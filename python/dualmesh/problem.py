"""The input of a simulation, built in Python as a YAML or TOML file builds it, and its solve.

The Python calls, a YAML file and a TOML file all build one input tree. A problem holds that
tree as the user wrote it, with units in text. `problem.solve` validates the tree, lowers it
onto the engine, solves it and returns the results.
"""

import copy
import os
from pathlib import Path

import numpy

from dualmesh.adaptivity import _Hierarchy
from dualmesh.input import schema
from dualmesh.input.build import (
    add_postprocessors,
    build_problem,
    stresses_at_points,
    transient_options,
)
from dualmesh.input.read import read as read_file
from dualmesh.input.read import write_TOML, write_YAML
from dualmesh.input.units import input_error
from dualmesh.results import results

# The fields that the output files hold in this version: the fields that the physics solve for.
written_fields = ("temperature", "displacement", "velocity", "pressure")


def table(file: str | None = None, **columns) -> dict:
    """A table value of a property or of a boundary flux constant.

    table("zircaloy_conductivity.csv") reads a CSV file whose header names each column with
    its unit. table(temperature=("K", [300, 600]), Young_modulus=("GPa", [96, 83])) gives the
    columns inline, each as (unit, values). The column named after the property holds its
    values, and every other column is an input.
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
        self._directory = os.getcwd()  # Relative mesh and table paths resolve against it
        sections = {"mesh": copy.deepcopy(mesh)}
        if method is not None:  # The tree holds only what the user chose
            sections["method"] = method
        self._set(**sections)

    # Building the tree

    def physics(self, name: str, **options) -> None:
        """Adds a physics, such as heat_conduction, with its own options and an optional method."""
        self._merge("physics", {name: options})

    def properties(self, **values) -> None:
        """Sets properties by standard name: a constant, a formula, a table, or a mapping of regions."""
        self._merge("properties", values)

    def initial_conditions(self, **values) -> None:
        """The value of each field at the start, by standard name, for example temperature="293 K"."""
        self._merge("initial_conditions", values)

    def boundary_conditions(self, boundary: str, **values) -> None:
        """The conditions on one named boundary, keyed by the quantity each one sets."""
        conditions = self._tree.get("boundary_conditions", {})
        self._merge("boundary_conditions", {boundary: {**conditions.get(boundary, {}), **values}})

    def gravity(self, components: list) -> None:
        """The gravitational acceleration, one quantity per direction of space, for example
        ["0 m/s^2", "-9.81 m/s^2"]. Solids and flows take its force rho g."""
        self._set(gravity=list(components))

    def time(self, **values) -> None:
        """Makes the analysis transient: end_time, initial_time_step and the step control."""
        self._set(time=dict(values))

    def mesh_adaptivity(self, **values) -> None:
        """Adapts the mesh to the error indicator of a field: indicator, field, refine_fraction,
        coarsen_fraction, interval and maximum_refinement_level."""
        self._set(mesh_adaptivity=dict(values))

    def postprocessor(self, name: str, **definition) -> None:
        """Adds a named scalar output, for example maximum="temperature"."""
        self._merge("postprocessors", {name: definition})

    def output(self, file: str, interval: int, fields: list[str]) -> None:
        """Writes the fields every interval-th accepted time step and at the end of the run.

        The files are <file>_<index>.vtu with a <file>.pvd collection, which ParaView opens as one
        time series, and <file>.csv with the postprocessors, a unit in each header.
        """
        self._set(output={"file": file, "interval": interval, "fields": list(fields)})

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
        tree = self.to_tree()
        schema.validate(tree, self._directory)
        folder = Path(path).resolve().parent

        def relative(file: str) -> str:
            if os.path.isabs(file):
                return file
            return os.path.relpath(os.path.join(self._directory, file), folder)

        return with_paths(tree, relative)

    @staticmethod
    def from_file(path: str) -> "problem":
        """Reads a problem from a YAML or TOML file. Relative paths in it resolve against its folder."""
        tree = read_file(path)
        directory = str(Path(path).resolve().parent)
        schema.validate(tree, directory)
        case = problem.__new__(problem)
        case._tree = tree
        case._directory = directory
        return case

    # Solving

    def solve(self, report: str = "summary") -> results:
        """Solves the problem: steady without a time section, transient with one."""
        validated = schema.validate(self._tree, self._directory)
        options = _solver_options(validated.get("solver", {}))
        if "mesh_adaptivity" in validated and "time" not in validated:
            engine, variables = _adapt(validated, report, options)
        else:
            engine, _, variables = build_problem(validated)
        coordinate_system = validated["mesh"]["coordinate_system"]
        dimension = engine.mesh.dimension
        units_of, stress_requests = add_postprocessors(
            engine, validated.get("postprocessors", {}), variables, dimension, coordinate_system
        )
        writer = _writer(validated.get("output"), engine, variables, self._directory)
        times = []

        def accepted(time, _engine):
            times.append(float(time))
            writer.step(time)

        history = None
        if "time" in validated and "mesh_adaptivity" in validated:
            engine, history = _adapt_in_time(
                validated, engine, variables, writer, accepted, report, options
            )
        elif "time" in validated:
            engine.set_time_step_callback(accepted)
            engine.solve_transient(**transient_options(validated["time"]), report=report, **options)
        else:
            engine.solve(report=report, **options)
            accepted(0.0, engine)
        stresses = stresses_at_points(engine, stress_requests, variables, validated)
        if history is None:
            history = engine.postprocessor_values()
        writer.finish(units_of, stresses, history)
        # The fields of the whole mesh, on every process of a distributed run.
        fields = {
            name: numpy.column_stack([engine.gathered_values(variable) for variable in names])
            for name, names in variables.items()
        }
        coordinates = engine.gathered_node_coordinates()[:, :dimension]
        postprocessors = {name: numpy.asarray(history[name]) for name in units_of}
        output_times = list(history["time"]) if units_of else times
        # A stress at a point is computed once, after a steady solve.
        postprocessors.update({name: numpy.asarray([value]) for name, value in stresses.items()})
        return results(
            times,
            fields,
            dict.fromkeys(fields, coordinates),
            output_times,
            postprocessors,
            {**units_of, **dict.fromkeys(stresses, "Pa")},
        )


def _bulk(squared, fraction):
    """The smallest set of cells whose squared indicators sum to the fraction of their total, the
    largest first (Dörfler marking)."""
    marked = numpy.zeros(squared.size, dtype=bool)
    total = squared.sum()
    if fraction <= 0.0 or total <= 0.0:
        return marked
    order = numpy.argsort(squared)[::-1]
    count = int(numpy.searchsorted(numpy.cumsum(squared[order]), fraction * total) + 1)
    marked[order[:count]] = True
    return marked


def _coarsening(squared, fraction, refine, coarsenable):
    """The largest set of refined cells outside ``refine`` whose squared indicators sum to at most
    the fraction of the total, the smallest first."""
    marked = numpy.zeros(squared.size, dtype=bool)
    candidates = numpy.flatnonzero(~refine & coarsenable)
    if fraction <= 0.0 or candidates.size == 0:
        return marked
    order = candidates[numpy.argsort(squared[candidates])]
    count = int(numpy.searchsorted(numpy.cumsum(squared[order]), fraction * squared.sum(), "right"))
    marked[order[:count]] = True
    return marked


def _check_adaptivity_field(settings: dict, variables: dict) -> None:
    """The field of mesh_adaptivity is one that the physics solve for."""
    if settings["field"] not in variables:
        raise input_error(
            f"mesh_adaptivity.field: '{settings['field']}' is not a field of the physics, "
            f"which solve for {', '.join(sorted(variables))}."
        )


def _adapt(validated: dict, report: str, options: dict):
    """The steady adaptive loop: solve, estimate, mark, adapt, until no cell below the maximum
    refinement level is marked to refine.  Each solve starts from the solution of the last mesh,
    carried to the new one.  It returns the engine problem on the last mesh and its fields."""
    from dualmesh import _core
    from dualmesh.input.build import build_mesh

    settings = validated["mesh_adaptivity"]
    hierarchy = _Hierarchy(build_mesh(validated["mesh"]))
    max_level = settings["maximum_refinement_level"]
    start = {}
    # Refinement and coarsening may undo each other, so the cycles are bounded.
    for cycle in range(4 * hierarchy.mesh().dimension * max_level + 1):
        engine, _, variables = build_problem(validated, hierarchy.mesh())
        _check_adaptivity_field(settings, variables)
        for name, values in start.items():
            engine.set_gathered_values(name, values)
        engine.solve(report="none", **options)
        # Every process holds the same hierarchy of the whole mesh, so the indicators and the
        # values are those of the whole mesh, and every process adapts it the same way.
        squared = sum(
            engine.gathered_error_indicator(name) ** 2 for name in variables[settings["field"]]
        )
        refine = _bulk(squared, settings["refine_fraction"])
        if report != "none" and _core.mpi_rank() == 0:
            print(
                f"  adaptivity cycle {cycle}: {hierarchy.mesh().num_elements} cells, "
                f"estimated error {numpy.sqrt(squared.sum()):.6g}",
                flush=True,
            )
        if not (refine & hierarchy.refinable(max_level)).any():
            break
        coarsen = _coarsening(
            squared, settings["coarsen_fraction"], refine, hierarchy.coarsenable()
        )
        before = {
            name: engine.gathered_values(name) for names in variables.values() for name in names
        }
        if not hierarchy.adapt(refine, coarsen, max_level):
            break
        start = {name: hierarchy.transfer(values) for name, values in before.items()}
    return engine, variables


def _adapt_in_time(validated, engine, variables, writer, accepted, report, options):
    """The transient adaptive loop: run interval accepted steps, adapt the mesh to the indicators
    of the solution reached, and go on from that time on the new mesh, with the solution carried
    over and the step that the stepper chose.  It returns the engine problem on the last mesh and
    the history of the post-processors of all the meshes."""
    from dualmesh.input.build import build_mesh

    settings = validated["mesh_adaptivity"]
    _check_adaptivity_field(settings, variables)
    time_options = transient_options(validated["time"])
    end_time = time_options["end_time"]
    hierarchy = _Hierarchy(build_mesh(validated["mesh"]))
    max_level = settings["maximum_refinement_level"]
    coordinate_system = validated["mesh"]["coordinate_system"]
    start_time, time_step = 0.0, time_options["time_step"]
    history = {}
    while True:
        engine.set_time_step_callback(accepted)
        result = engine.solve_transient(
            **{**time_options, "start_time": start_time, "time_step": time_step},
            max_steps=settings["interval"],
            report=report,
            **options,
        )
        # The post-processors of this mesh, after the time where those of the last one ended.
        reached = engine.postprocessor_values()
        last = history["time"][-1] if history.get("time") else -numpy.inf
        keep = [k for k, t in enumerate(reached.get("time", [])) if t > last]
        for name, values in reached.items():
            history.setdefault(name, []).extend(values[k] for k in keep)
        start_time = float(engine.time)
        if start_time >= end_time - 1e-12 * abs(end_time):
            break
        time_step = result.next_time_step
        squared = sum(
            engine.gathered_error_indicator(name) ** 2 for name in variables[settings["field"]]
        )
        refine = _bulk(squared, settings["refine_fraction"]) & hierarchy.refinable(max_level)
        coarsen = _coarsening(
            squared, settings["coarsen_fraction"], refine, hierarchy.coarsenable()
        )
        state = {
            name: engine.gathered_values(name) for names in variables.values() for name in names
        }
        # The velocity of a second-order equation is a state of the step too: the new problem
        # goes on from the velocity reached, not from that of the initial condition.
        velocity = engine.gathered_velocities()
        if hierarchy.adapt(refine, coarsen, max_level):
            state = {name: hierarchy.transfer(v) for name, v in state.items()}
            velocity = {name: hierarchy.transfer(v) for name, v in velocity.items()}
            engine, _, variables = build_problem(validated, hierarchy.mesh())
            engine.set_gathered_velocities(velocity)
            add_postprocessors(
                engine,
                validated.get("postprocessors", {}),
                variables,
                engine.mesh.dimension,
                coordinate_system,
            )
            writer.engine = engine
        for name, values in state.items():
            engine.set_gathered_values(name, values)
    return engine, history


def _solver_options(solver: dict) -> dict:
    options = {}
    if "relative_tolerance" in solver:
        options["relative_tolerance"] = solver["relative_tolerance"]
    if "PETSc_options" in solver:
        options["petsc_options"] = dict(solver["PETSc_options"])
    return options


class _no_output:
    engine = None

    def step(self, time) -> None:
        pass

    def finish(self, units_of, stresses, history) -> None:
        pass


class _files:
    """The output files: the fields every interval-th accepted step and at the end, and the postprocessors."""

    def __init__(self, output: dict, engine, variables: dict, directory: str):
        for index, name in enumerate(output["fields"]):
            if name not in written_fields or name not in variables:
                raise input_error(
                    f"output.fields[{index}]: the files hold the fields that the physics solve for in this "
                    f"version, here {', '.join(variable for variable in written_fields if variable in variables)}."
                )
        self.base = os.path.join(directory, output["file"])
        self.interval = output["interval"]
        self.engine = engine
        self.names = [variable for name in output["fields"] for variable in variables[name]]
        self.count = 0
        self.written: list[tuple[float, str]] = []

    def write(self, time) -> None:
        path = f"{self.base}_{len(self.written):04d}.vtu"
        self.engine.write_vtu(path, fields=self.names)
        self.written.append((float(time), path))

    def step(self, time) -> None:
        if self.count % self.interval == 0:
            self.write(time)
        self.count += 1
        self.last_time = float(time)

    def finish(self, units_of, stresses, history) -> None:
        if not self.written or self.written[-1][0] != self.last_time:
            self.write(self.last_time)
        datasets = "\n".join(
            f'    <DataSet timestep="{time!r}" file="{os.path.basename(path)}"/>'
            for time, path in self.written
        )
        Path(self.base + ".pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
            f"{datasets}\n  </Collection>\n</VTKFile>\n"
        )
        if units_of or stresses:
            history = history if units_of else {"time": [self.last_time]}
            headers = ["time [s]"] + [f"{name} [{unit}]" for name, unit in units_of.items()]
            headers += [f"{name} [Pa]" for name in stresses]
            columns = [history["time"], *(history[name] for name in units_of)]
            columns += [[value] * len(history["time"]) for value in stresses.values()]
            lines = [",".join(headers)] + [
                ",".join(repr(float(value)) for value in row) for row in zip(*columns)
            ]
            Path(self.base + ".csv").write_text("\n".join(lines) + "\n")


def _writer(output, engine, variables, directory):
    return _no_output() if output is None else _files(output, engine, variables, directory)


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
