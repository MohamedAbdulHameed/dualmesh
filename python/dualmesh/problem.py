# SPDX-License-Identifier: LGPL-2.1-or-later
"""The :class:`Problem` class: variables, objects, solvers, and results."""

from __future__ import annotations

import difflib
import re
import time
from collections.abc import Iterable, Sequence

import numpy as np

from . import _core
from ._core import InputError
from .console import report_level
from .output import Output

SolveResult = _core.SolveResult


def _solve_result_tables(self) -> dict:
    """``newton``: one row per Newton iteration."""
    from .tables import Table

    return {"newton": Table(["load_step", "load_factor", "iteration", "residual_norm", "step_norm"], None, [[r.load_step, r.load_factor, r.iteration, r.residual_norm, r.step_norm] for r in self.history], title="Newton iterations")}


def _solve_result_to_dict(self) -> dict:
    """Whether the solve converged, its iteration counts and the history of
    its Newton iterations."""
    history = [dict(load_step=r.load_step, load_factor=r.load_factor, iteration=r.iteration, residual_norm=r.residual_norm, step_norm=r.step_norm) for r in self.history]
    return {"study": "solve", "converged": bool(self.converged), "total_iterations": int(self.total_iterations), "linear_iterations": int(self.linear_iterations), "time_steps": int(self.time_steps), "rejected_steps": int(self.rejected_steps), "history": history}


def _solve_result_summary(self) -> str:
    """The outcome of the solve, as a table."""
    from .console import solve_report

    return solve_report(self)


def _solve_result_write_json(self, path) -> None:
    """Write :meth:`to_dict` to a JSON file."""
    from .parameters import write_json_numbers

    write_json_numbers(self.to_dict(), path)


def _solve_result_write_csv(self, path, table: str | None = None) -> None:
    """Write the Newton iterations (the table ``newton``) to a CSV file."""
    from .tables import ResultTables

    ResultTables.write_csv(self, path, table)


SolveResult.tables = property(_solve_result_tables)
SolveResult.to_dict = _solve_result_to_dict
SolveResult.summary = _solve_result_summary
SolveResult.write_json = _solve_result_write_json
SolveResult.write_csv = _solve_result_write_csv


def _as_function(value):
    """A number, an expression string, a compiled function or a callable, as
    something the extension accepts where it expects a function."""
    if isinstance(value, str):
        return _core.ParsedFunction(value)
    return value


def petsc_option_string(options) -> str:
    """PETSc options as the option string PETSc parses.

    ``options`` may already be a string (``"-ksp_type cg -pc_type gamg"``), a
    dictionary (``{"ksp_type": "cg", "pc_type": "gamg"}``, where a value of
    ``None`` or ``True`` gives a flag without a value), or ``None``."""
    if options is None:
        return ""
    if isinstance(options, str):
        return options
    parts = []
    for key, value in options.items():
        name = key if key.startswith("-") else "-" + key
        if value is None or value is True:
            parts.append(name)
        elif value is False:
            continue
        else:
            parts.append(f"{name} {value}")
    return " ".join(parts)


def _solver_options(**kwargs) -> _core.SolverOptions:
    options = _core.SolverOptions()
    known = {
        "nonlinear_solver",
        "max_iterations",
        "relative_tolerance",
        "absolute_tolerance",
        "step_tolerance",
        "relaxation",
        "line_search",
        "max_line_search_steps",
        "load_factors",
        "linear_solver",
        "preconditioner",
        "gmres_restart",
        "linear_tolerance",
        "amg_strength_threshold",
        "linear_max_iterations",
        "petsc_options",
        "error_on_divergence",
    }
    for key, value in kwargs.items():
        if key == "verbose":
            raise TypeError("Unknown solver option 'verbose'. The parameter report sets what a solve prints: report='full' (the default), 'summary' or 'none'.")
        if key == "_progress":
            options.verbose = bool(value)
            continue
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise TypeError(f"Unknown solver option '{key}'.{hint} Known options: {', '.join(sorted(known))}.")
        if key == "petsc_options":
            value = petsc_option_string(value)
        setattr(options, key, value)
    return options


#: The options of the linear solver of a distributed problem.
_DISTRIBUTED_LINEAR = ("linear_solver", "preconditioner", "subdomain_solver", "linear_tolerance", "linear_max_iterations", "petsc_options")


def _distributed_options(options: dict):
    """Split the solver options of a distributed problem into its linear
    solver settings and the options of the nonlinear solver."""
    linear = _core.DistributedOptions()
    rest = dict(options)
    solver = rest.pop("linear_solver", "automatic")
    linear.linear_solver = "bicgstab" if solver == "automatic" else solver
    linear.preconditioner = rest.pop("preconditioner", "two_level_schwarz")
    linear.subdomain_solver = rest.pop("subdomain_solver", "ilu")
    linear.linear_tolerance = float(rest.pop("linear_tolerance", 1e-10))
    linear.linear_max_iterations = int(rest.pop("linear_max_iterations", 5000))
    linear.petsc_options = petsc_option_string(rest.pop("petsc_options", None))
    linear.verbose = bool(rest.get("_progress", False))
    return linear, rest


class Problem:
    """A boundary value problem discretized on a primal mesh and its dual mesh.

    Parameters
    ----------
    mesh:
        The primal mesh of finite elements.
    method:
        The discretization.  ``"dmcdm"`` is the dual mesh control domain method
        (the default), and ``"fem"`` is the Galerkin finite element method.
        ``"hfvm"`` is the vertex-centred finite volume method, which uses the
        same control domains as the dual mesh method and two-point gradients at
        their interfaces (the half-control volume formulation of Reddy,
        Chapter 3).  ``"zfvm"`` is the cell-centred finite volume method, with
        one unknown per element and one per boundary face (the zero-thickness
        control volume formulation, and the layout used by OpenFOAM).  The rest
        of the problem definition is identical for all four, which makes them
        directly comparable.
    boundary_gradient:
        Only for ``"zfvm"``: ``"first_order"`` (the two-point difference
        between the cell and the boundary node, the default) or
        ``"second_order"`` (the one-sided quadratic of Eq. (3.2.14) of the
        book, through the boundary node and the two nearest cells).
    coordinates:
        ``"cartesian"``, ``"axisymmetric"`` (the integrals carry the factor
        :math:`2 \\pi r`, with :math:`r` the first coordinate), or
        ``"spherical"`` (factor :math:`4 \\pi r^2`, one-dimensional meshes).
    distributed:
        Whether the problem is split among the MPI processes.  The default,
        ``None``, splits it when the script runs on more than one process
        (``mpirun -n 4 python script.py``), so that one script runs serially
        and in parallel.  ``True`` takes the distributed path on one process
        as well, which gives the serial answer.  In a distributed problem
        every process defines the same physics and objects, each assembles
        the elements of its own part of the mesh, and :meth:`solve` takes the
        distributed linear solvers (see :meth:`solve`).
    partitioner:
        How a distributed problem splits the mesh: ``"graph"`` (the default,
        parts grown through the face connectivity), ``"metis"`` (METIS, when
        the extension has it) or ``"recursive_coordinate_bisection"``.  See
        :func:`dualmesh.partition_mesh`.
    overlap:
        Layers of elements by which each Schwarz subdomain of a distributed
        problem reaches into its neighbours.  Default 1, which halves the
        iteration count of the non-overlapping subdomains on the Poisson
        problem.  More overlap lowers the iteration count and raises the cost
        of each subdomain solve.
    """

    def __init__(self, mesh, method: str = "dmcdm", coordinates: str = "cartesian", boundary_gradient: str = "first_order", distributed: bool | None = None, partitioner: str = "graph", overlap: int = 1):
        if distributed is None:
            distributed = _core.mpi_size() > 1
        self._distributed = None
        if distributed:
            options = _core.DistributedOptions()
            options.partitioner = partitioner
            options.overlap = overlap
            self._distributed = _core.DistributedProblem(mesh, method, coordinates, options)
            self._problem = self._distributed.local()
            self._global_mesh = mesh
            mesh = self._problem.mesh
        else:
            self._problem = _core.Problem(mesh, method, coordinates)
        self._problem.set_boundary_gradient(boundary_gradient)
        self._mesh = mesh
        self._objects: list = []  # keeps Python-defined objects alive
        self._init_postprocessing()
        self._physics: dict = {}
        self._couplings: dict = {}
        self.method = method
        self.coordinates = coordinates
        self.boundary_gradient = boundary_gradient

    def set_num_threads(self, num_threads: int) -> None:
        """Number of threads used by the assembly loops.

        ``0`` means the OpenMP default, which is the value of the
        ``OMP_NUM_THREADS`` environment variable or, if that is unset, the
        number of cores.  The setting is a request: the assembly falls back to
        one thread when the library was built without OpenMP, when the mesh is
        too small for threading to pay for itself, or when any kernel, boundary
        condition, material or function of the problem is defined in Python,
        because calling back into the interpreter needs the global interpreter
        lock.  :meth:`effective_threads` reports the number actually used.
        """
        self._problem.set_num_threads(int(num_threads))

    def effective_threads(self) -> int:
        """The number of threads the assembly will actually use."""
        return self._problem.effective_threads()

    @property
    def thread_safe(self) -> bool:
        """False when some object of the problem is defined in Python, which
        forces the assembly onto one thread."""
        return self._problem.thread_safe()

    def _init_postprocessing(self) -> None:
        from .postprocessors import PostprocessorHistory

        self._postprocessors = PostprocessorHistory()
        self._user_step_callback = None
        self._domain_volume = None

    @property
    def is_cell_centered(self) -> bool:
        """True when the unknowns sit at cell centroids instead of mesh nodes."""
        return self._problem.is_cell_centered()

    def entity_points(self) -> np.ndarray:
        """Positions of the degrees of freedom, one row each, in the order of
        :meth:`values`.

        For the dual mesh, finite element and vertex-centred finite volume
        methods these are the mesh nodes.  For the cell-centred finite volume
        method they are the cell centroids first and then the boundary face
        centroids.
        """
        n = self._problem.num_entities()
        return np.asarray([self._problem.entity_point(i) for i in range(n)])

    def boundary_entities(self, boundary: str) -> list[int]:
        """Degree of freedom indices that carry the values of a boundary."""
        return list(self._problem.boundary_entities(boundary))

    # ---- definition ------------------------------------------------------
    def add_variable(self, name: str, block: Sequence[str] = (), initial_condition=None, order: str = "mesh", unit: str = "") -> int:
        """Add a nodal unknown and return its index.

        ``unit`` is the SI unit of the variable, which the CSV and VTU files
        write with its values.  Default none.  A physics gives the units of
        the variables that it makes.

        ``order`` is the polynomial order the variable is interpolated with:

        * ``"mesh"`` (the default): the order of the elements, linear on a
          linear mesh and quadratic on a quadratic one.
        * ``"first"``: linear on every element.  On a quadratic mesh only the
          corner nodes carry the variable, and the values reported at the other
          nodes are those of the linear field.  This is the pressure of the
          Taylor-Hood element (quadratic velocity, linear pressure), which is
          stable for incompressible flow without stabilisation.  It needs the
          finite element method (``method="fem"``).
        """
        index = self._problem.add_variable(name, list(block), initial_condition, order)
        if unit:
            self._problem.set_variable_unit(name, unit)
        return index

    def variable_unit(self, name: str) -> str:
        """The SI unit of a variable (an empty text when none was given)."""
        return self._problem.variable_unit(name)

    def num_active_dofs(self) -> int:
        """The number of unknowns that carry an equation.  It is smaller than
        the number of entities times the number of variables when a variable is
        restricted to some blocks, or when it is first order on a quadratic
        mesh (it then lives on the corner nodes only)."""
        return self._problem.num_active_dofs()

    def variable_order(self, name: str) -> str:
        """The order a variable was declared with: ``"mesh"`` or ``"first"``."""
        return self._problem.variable_order(name)

    def add_function(self, name: str, function) -> None:
        """Register a named function of ``(x, y, z, t)`` (or a constant)."""
        self._problem.add_function(name, function)

    def _add(self, adder, object_or_type, name, kwargs):
        if isinstance(object_or_type, str):
            # A matrix (a list of rows of numbers) is passed row by row.
            kwargs = {k: [float(x) for row in v for x in row] if isinstance(v, (list, tuple)) and v and all(isinstance(r, (list, tuple, np.ndarray)) for r in v) else v for k, v in kwargs.items()}
            return self._problem.add_object(object_or_type, name or "", **kwargs)
        if kwargs:
            raise TypeError("Extra parameters are not accepted when adding an object instance.")
        self._objects.append(object_or_type)
        adder(object_or_type)
        return object_or_type

    def add_kernel(self, kernel, name: str | None = None, **parameters):
        """Add a kernel by registered type name, or an object built in Python."""
        return self._add(self._problem.add_kernel, kernel, name, parameters)

    def add_boundary_condition(self, condition, name: str | None = None, **parameters):
        """Add a boundary condition (essential or natural).

        When ``boundary`` is not given, the name of the condition is taken as
        its boundary, so that a condition named after a side set acts on that
        side set::

            problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="u", value=0.0)

        Several conditions on one boundary need different names and an
        explicit ``boundary``."""
        if isinstance(condition, str):
            if "boundary" not in parameters and name:
                mesh = self._mesh
                known = set(mesh.sideset_names()) | set(mesh.nodeset_names())
                if name in known:
                    parameters["boundary"] = [name]
                elif re.search(r"^\s+boundary \([^)]*required", _core.describe_object(condition), re.M):
                    raise ValueError(f"Boundary condition '{name}': give 'boundary', or name the condition after a boundary of the mesh ({', '.join(sorted(known)) or 'none'}).")
            category = _core.object_category(condition)
            if category == "nodal_boundary_condition":
                return self._problem.add_object(condition, name or "", **parameters)
            if category == "boundary_condition":
                return self._problem.add_object(condition, name or "", **parameters)
            raise ValueError(f"'{condition}' is a {category}, not a boundary condition.")
        if isinstance(condition, _core.NodalBC):
            return self._add(self._problem.add_nodal_bc, condition, name, parameters)
        return self._add(self._problem.add_integrated_bc, condition, name, parameters)

    def add_property(self, property_type, name: str | None = None, **parameters):
        """Add a property object: an object that computes named properties at
        the integration points of its blocks (e.g., a thermal conductivity,
        a stress or the cross sections of a region), which the physics and
        the kernels read by name::

            problem.add_property("constant_property", "copper", block=["bar"], property_names=["thermal_conductivity"], property_values=[400.0])
        """
        return self._add(self._problem.add_property, property_type, name, parameters)

    def add_point_source(self, source="point_source", name: str | None = None, **parameters):
        """Add a concentrated nodal source (point force or point heat source)."""
        return self._add(self._problem.add_nodal_load, source, name, parameters)

    def add_physics(self, physics: str, name: str | None = None, **parameters):
        """Add a set of equations named in physical terms (see
        :mod:`dualmesh.physics` and ``dualmesh list --category physics``).

        The physics creates its variables now and generates its kernels and
        materials when the problem is initialised.  It returns the physics,
        whose :meth:`~dualmesh.physics.Physics.add_boundary_condition` and
        :meth:`~dualmesh.physics.Physics.add_kernel` add conditions and terms
        to its equations::

            heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
            heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
        """
        from . import physics as _physics

        name = name or physics
        if name in self._physics or name in self._couplings:
            raise _physics.InputError(f"The problem already has a physics or coupling named '{name}'.")
        instance = _physics.create(physics, name, parameters)
        if not isinstance(instance, _physics.Physics):
            raise _physics.InputError(f"'{physics}' is a coupling. Add it with add_coupling.")
        instance._attach(self)
        self._physics[name] = instance
        return instance

    def add_coupling(self, coupling: str, name: str | None = None, **parameters):
        """Couple two physics of the problem, e.g., ``thermal_expansion``,
        ``Boussinesq_buoyancy`` or ``heat_convection`` (``dualmesh list
        --category coupling``)::

            problem.add_coupling("thermal_expansion", "expansion", heat_transfer="heat", solid_mechanics="solid", thermal_expansion_coefficient=1.2e-5, stress_free_temperature=293.15)
        """
        from . import physics as _physics

        name = name or coupling
        if name in self._physics or name in self._couplings:
            raise _physics.InputError(f"The problem already has a physics or coupling named '{name}'.")
        instance = _physics.create(coupling, name, parameters)
        if not isinstance(instance, _physics.Coupling):
            raise _physics.InputError(f"'{coupling}' is a physics. Add it with add_physics.")
        if any(p._built for p in self._physics.values()):
            raise _physics.InputError(f"Coupling '{name}': add the couplings before the problem is solved or initialised.")
        instance._apply(self)
        self._couplings[name] = instance
        return instance

    def _build_physics(self) -> None:
        """Generate the objects of the physics and couplings not built yet."""
        new = [p for p in getattr(self, "_physics", {}).values() if not p._built]
        if not new:
            return
        for physics in new:
            physics._build(self)
            physics._built = True
        for coupling in self._couplings.values():
            if not getattr(coupling, "_built", False):
                coupling._build(self)
                coupling._built = True

    def initialize(self) -> None:
        """Resolve all objects (called automatically by :meth:`solve`)."""
        self._build_physics()
        self._problem.initialize()

    # ---- solving ---------------------------------------------------------
    def solve(self, report: str = "full", output=None, **options) -> SolveResult:
        r"""Solve the steady problem.

        ``report`` sets what the solve prints: ``"full"`` (the default: the
        problem with every parameter, the defaults marked, the Newton
        iterations and the result), ``"summary"`` (one line that names the
        solve, then the result) or ``"none"``.  ``output`` is an
        :class:`~dualmesh.Output` group: the solve writes the fields once, at
        the end.  Default None: the solve writes no file.

        Keyword arguments are solver options: ``nonlinear_solver`` (``"newton"``,
        ``"picard"``, or ``"linear"``), ``max_iterations``, ``relative_tolerance``,
        ``absolute_tolerance``, ``step_tolerance``, ``relaxation`` (the
        acceleration parameter of direct iteration), ``line_search``
        (``"none"``, the default, or ``"backtracking"``, which halves a Newton
        step until the residual norm falls) and ``max_line_search_steps``
        (default 10), ``load_factors`` (load
        stepping), ``linear_solver`` (``"automatic"``, the default, ``"lu"``,
        ``"bicgstab"``, ``"gmres"``, ``"cg"`` or ``"petsc"``),
        ``preconditioner`` (``"ilu"``, the default, ``"ilut"``, ``"amg"``,
        ``"jacobi"``, ``"none"``, or ``"pressure_mass_schur"`` for
        pressure-velocity flow), ``amg_strength_threshold``,
        ``linear_tolerance``, ``linear_max_iterations``, ``gmres_restart``,
        ``petsc_options`` (PETSc's options, as a string or a dictionary, for
        ``linear_solver="petsc"``) and ``error_on_divergence``.
        :doc:`/user_guide/solving` explains every option and its default.

        ``"automatic"`` factorises the system directly where that is cheap,
        which is always in one dimension, up to :math:`10^5` unknowns in two
        and up to a few thousand in three.  Larger systems are solved by the
        conjugate gradient method or BiCGSTAB preconditioned by algebraic
        multigrid, then by BiCGSTAB with an incomplete LU factorisation, and
        then by the direct solver, each one taken when the previous one does
        not converge.  The factorisation and the multigrid hierarchy are kept
        while the matrix does not change.

        In a distributed problem (see the parameter ``distributed`` of the
        class) the linear systems are solved by distributed Krylov methods:
        ``linear_solver`` is ``"bicgstab"`` (the default for any matrix),
        ``"cg"`` (symmetric positive definite matrices) or ``"petsc"`` (PETSc's
        solvers, with ``petsc_options``), and ``preconditioner`` is
        ``"two_level_schwarz"`` (the default: restricted additive Schwarz on
        overlapping subdomains with a coarse level of one unknown per
        subdomain and variable, whose iteration count does not grow with the
        number of processes), ``"additive_schwarz"`` (without the coarse
        level) or ``"jacobi"``.  ``subdomain_solver`` is ``"ilu"`` (the
        default) or ``"lu"`` for the subdomain problems.  :doc:`/theory/parallel`
        gives the method and the iteration counts.
        """
        level = report_level(report, "Problem.solve")
        output = self._check_output(output, steady=True)
        self._build_physics()
        self._report_start(level, "steady solve")
        start = time.perf_counter()
        options["_progress"] = level == "full"
        if self._distributed is not None:
            linear, options = _distributed_options(options)
            self._distributed.set_linear_solver(linear)
            result = self._distributed.solve_steady(_solver_options(**options))
        else:
            result = self._problem.solve_steady(_solver_options(**options))
        if self._postprocessors.postprocessors:
            self._postprocessors.evaluate(self, self.time)
        files = self._write_fields(output) if output is not None else []
        self._report_end(level, result, time.perf_counter() - start, files)
        return result

    # ---- reports and files of the solves ------------------------------------
    def _report_start(self, level: str, study: str) -> None:
        if level == "full":
            from .console import header

            print(header(study))
            print(self.summary(), flush=True)
        elif level == "summary":
            print(f"dualmesh {study}: {self.num_active_dofs()} unknowns, method {self.method}", flush=True)

    def _report_end(self, level: str, result, wall_time: float, files: Sequence[str] = ()) -> None:
        if level == "none":
            return
        from .console import solve_report

        # The progress lines of the library come before the result.
        _core.flush_output()

        print(solve_report(result, wall_time))
        if self._postprocessors.postprocessors:
            print("Post-processors\n" + self.postprocessor_table())
        if level == "full":
            for path in files:
                print(f"  wrote {path}")

    def _check_output(self, output, steady: bool):
        """Check an Output group of a solve."""
        if output is None:
            return None
        from .output import Output

        if not isinstance(output, Output):
            raise InputError(f"The output of a solve is a dualmesh.Output group, not {type(output).__name__}.")
        if steady and (output.times is not None or output.interval is not None):
            raise InputError("Output: a steady solve writes the fields once, at the end. Remove times and interval.")
        if output.fields is not None:
            names = [self._problem.variable_name(i) for i in range(self._problem.num_variables)]
            for name in output.fields:
                if name not in names:
                    close = difflib.get_close_matches(name, names, n=1)
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise InputError(f"Output: the field '{name}' is not a variable.{hint} The variables are: {', '.join(names)}.")
        if self._distributed is not None and "csv" in output.formats:
            raise InputError("Output: a distributed problem writes the vtu format only. Remove csv from formats.")
        return output

    def _write_fields(self, output, index: int | None = None) -> list[str]:
        """Write the fields of the Output group in its formats.  ``index`` is
        the number of the output time of a transient study."""
        suffix = "" if index is None else f"_{index:05d}"
        fields = list(output.fields or ())
        files = []
        if "vtu" in output.formats:
            path = output.path(suffix + ".vtu")
            self.write_vtu(path, fields=fields)
            files.append(path[:-4] + ".pvtu" if self._distributed is not None and self.num_ranks > 1 else path)
        if "csv" in output.formats:
            path = output.path(suffix + ".csv")
            self.write_csv(path, variables=fields)
            files.append(path)
        return files

    def solve_transient(
        self,
        end_time: float,
        time_step: float,
        start_time: float = 0.0,
        implicitness: float = 1.0,
        output=None,
        report: str = "full",
        time_stepper: str = "fixed",
        min_time_step: float = 0.0,
        max_time_step: float = 0.0,
        growth_factor: float = 2.0,
        cutback_factor: float = 0.5,
        error_tolerance: float = 1.0e-3,
        optimal_iterations: int = 4,
        iteration_window: int = 2,
        max_rejected_steps: int = 10,
        **options,
    ) -> SolveResult:
        r"""March the solution forward in time with the theta method.

        The theta method weights the steady part of the residual between the
        old state and the new one,

        .. math::

            R_{\text{time}}(U^{n+1})
            + \theta R_{\text{steady}}(U^{n+1})
            + (1 - \theta) R_{\text{steady}}(U^{n}) = 0,

        where :math:`\theta` is the ``implicitness``.  ``implicitness=1`` is
        the backward Euler method, which is unconditionally stable and
        first-order accurate.  ``implicitness=0.5`` is the Crank-Nicolson, or
        midpoint, rule, which is unconditionally stable and second-order
        accurate, and which can oscillate after a sharp transient.
        ``implicitness=0`` is the forward Euler method, which is explicit in
        the steady terms and is stable only below a critical step.

        Parameters
        ----------
        end_time, time_step, start_time:
            The interval to cover and the step to take.  With an adaptive
            stepper ``time_step`` is the first step.
        implicitness:
            The weight :math:`\theta` above.  Default 1, the backward Euler
            method, which is stable for every step.
        output:
            An :class:`~dualmesh.Output` group: the files, the output times,
            the fields and the formats.  The time steps land on the output
            times.  A transient study also writes a ParaView collection
            (``.pvd``) of its ``.vtu`` files, which ParaView opens as one time
            series.  Default None: the study writes no file.
        report:
            What the study prints: ``"full"`` (the default: the problem with
            every parameter, the defaults marked, a line for each time step
            and the result), ``"summary"`` (one line that names the study,
            then the result) or ``"none"``.
        time_stepper:
            ``"fixed"`` keeps the step, shortening only the last one so that
            the run lands exactly on ``end_time``.

            ``"error"`` chooses the step from an estimate of the local
            truncation error.  Each interval is advanced twice, once with one
            step and once with two half steps, and the difference between the
            two answers estimates the error of the coarse one by Richardson
            extrapolation.  A step whose relative error exceeds
            ``error_tolerance`` is discarded and retried with a smaller step,
            and an accepted step is followed by the largest step the estimate
            allows.  The answer that is kept is the accurate one, from the two
            half steps.  The estimate costs three nonlinear solves per accepted
            step, so use this when accuracy in time is what matters.

            ``"iteration"`` chooses the step from how hard the nonlinear solver
            worked.  A step that converged in fewer than
            ``optimal_iterations - iteration_window`` iterations is followed by
            a larger one, a step that needed more than
            ``optimal_iterations + iteration_window`` by a smaller one, and a
            step that failed to converge is discarded and retried.  It costs
            nothing beyond the solve and suits a problem whose difficulty lies
            in the nonlinearity.
        min_time_step, max_time_step:
            Bounds on the step.  A run that has to go below ``min_time_step``
            is reported as a failure.  Zero means ``time_step`` divided by one
            million, and the whole interval, respectively.
        growth_factor, cutback_factor:
            The most the step may grow between accepted steps, and the factor
            applied after a rejected one.
        error_tolerance:
            The target for the relative local error of one step.
        optimal_iterations, iteration_window:
            The iteration count the ``"iteration"`` stepper aims for, and the
            half-width of the band around it inside which the step is left
            alone.
        max_rejected_steps:
            How many times in a row a step may be rejected before the run is
            declared a failure.
        **options:
            Passed to the nonlinear solver of every step (see :meth:`solve`).

        Returns
        -------
        SolveResult
            Besides the usual fields, ``time_steps`` counts the accepted steps,
            ``rejected_steps`` the discarded ones, and ``step_history`` holds
            the time reached and the step taken for each accepted step.
        """
        transient = _core.TransientOptions()
        transient.start_time = start_time
        transient.end_time = end_time
        transient.dt = time_step
        transient.theta = implicitness
        transient.time_stepper = time_stepper
        transient.dt_min = min_time_step
        transient.dt_max = max_time_step
        transient.growth_factor = growth_factor
        transient.cutback_factor = cutback_factor
        transient.error_tolerance = error_tolerance
        transient.optimal_iterations = optimal_iterations
        transient.iteration_window = iteration_window
        transient.max_rejected_steps = max_rejected_steps
        level = report_level(report, "Problem.solve_transient")
        self._build_physics()
        output = self._check_output(output, steady=False)
        files: list[str] = []
        csv_output = None
        if output is not None:
            transient.output_times = [float(t) for t in output.output_times(start_time, end_time)]
            if "vtu" in output.formats:
                transient.output_file_base = output.path("")
                transient.output_fields = list(output.fields or ())
            if "csv" in output.formats:
                # The library calls this writer at each output time, after
                # the boundary values of the start are applied.
                csv_output = Output(directory=output.directory, file_base=output.file_base, fields=output.fields, formats=("csv",))
                self._problem.set_output_callback(lambda index: files.extend(self._write_fields(csv_output, index)))
        self._report_start(level, "transient solve")
        start = time.perf_counter()
        if self._postprocessors.postprocessors:
            self.time = start_time
            self._postprocessors.evaluate(self, start_time)
        options["_progress"] = level == "full"
        try:
            if self._distributed is not None:
                linear, options = _distributed_options(options)
                self._distributed.set_linear_solver(linear)
                result = self._distributed.solve_transient(transient, _solver_options(**options))
            else:
                result = self._problem.solve_transient(transient, _solver_options(**options))
        finally:
            if csv_output is not None:
                self._problem.set_output_callback(None)
        if output is not None and "vtu" in output.formats:
            extension = "pvtu" if self._distributed is not None and self.num_ranks > 1 else "vtu"
            files.append(output.write_collection(transient.output_times, extension))
        self._report_end(level, result, time.perf_counter() - start, files)
        return result

    def solve_eigenvalue(self, method: str = "krylov", tolerance: float = 1.0e-10, max_iterations: int = 2000, normalization: float = 1.0, report: str = "full", output=None):
        """Compute the effective multiplication factor and the fundamental
        mode of the neutron_diffusion physics of the problem (see
        :mod:`dualmesh.eigenvalue`).

        ``method`` is ``"krylov"`` (the Arnoldi method, the default, which
        converges in far fewer operator applications than the power
        iteration when the dominance ratio is close to one) or ``"power"``
        (the power iteration).  ``tolerance`` is the relative tolerance on
        the eigenvalue.  The fluxes are scaled so that the total production
        of fission neutrons, the integral of the sum over the groups of
        :math:`\\nu\\Sigma_{f,g} \\phi_g`, equals ``normalization``.  Returns
        an :class:`~dualmesh.eigenvalue.EigenvalueResult`.

        ``report`` sets what the study prints: ``"full"`` (the default: the
        problem with every parameter, the defaults marked, and the result),
        ``"summary"`` (one line that names the study, then the result) or
        ``"none"``.  ``output`` is an :class:`~dualmesh.Output` group: the
        study writes the fluxes once, at the end.  Default None: no file."""
        from .eigenvalue import solve_eigenvalue

        level = report_level(report, "Problem.solve_eigenvalue")
        output = self._check_output(output, steady=True)
        self._build_physics()
        if level == "full":
            from .console import header

            print(header("eigenvalue study"))
            print(self.summary())
        elif level == "summary":
            print(f"dualmesh eigenvalue study: {self.num_active_dofs()} unknowns, method {self.method}, eigenvalue solver {method}")
        result = solve_eigenvalue(self, method=method, tolerance=tolerance, max_iterations=max_iterations, normalization=normalization)
        files = self._write_fields(output) if output is not None else []
        if level != "none":
            print(result.summary())
        if level == "full":
            for path in files:
                print(f"  wrote {path}")
        return result

    def set_time_step_callback(self, callback) -> None:
        """Call ``callback(time, problem)`` after every converged time step
        (after the post-processors have been evaluated)."""
        self._user_step_callback = callback
        self._install_step_callback()

    def _install_step_callback(self) -> None:
        user = self._user_step_callback
        if not self._postprocessors.postprocessors:
            if user is not None:
                self._problem.set_time_step_callback(user)
            return

        def step(time, core_problem):
            self._postprocessors.evaluate(self, time)
            if user is not None:
                user(time, core_problem)

        self._problem.set_time_step_callback(step)

    # ---- post-processors -------------------------------------------------------
    def add_postprocessor(self, postprocessor_type: str, name: str, **parameters):
        """Add a post-processor, a scalar computed from the solution after every
        steady solve and every accepted time step: ``variable_integral``,
        ``variable_average``, ``point_value``, ``nodal_extreme_value``,
        ``reaction`` or ``boundary_flux_integral`` (see
        :mod:`dualmesh.postprocessors`)."""
        from .postprocessors import create

        if self._distributed is not None:
            raise ValueError("Post-processors are not available in a distributed problem yet. Compute the quantity from gathered_values, or run the problem on one process.")
        pp = create(postprocessor_type, name, **parameters)
        self._postprocessors.add(pp)
        self._install_step_callback()
        return pp

    def postprocessor_values(self) -> dict:
        """The post-processors' history: ``{"time": array, name: array, ...}``."""
        return self._postprocessors.as_arrays()

    def postprocessor_table(self) -> str:
        """The post-processors' history as aligned text."""
        return self._postprocessors.table()

    def write_postprocessor_csv(self, filename: str) -> None:
        """Write the post-processors' history to a CSV file, one row per
        evaluation, with a header row of names."""
        self._postprocessors.write_csv(filename)

    def domain_volume(self) -> float:
        """The volume (area, length) of the domain, with the coordinate factor
        of the problem (2 pi r for axisymmetric, 4 pi r^2 for spherical)."""
        if self._domain_volume is None:
            probe = Problem(self._mesh, method="fem", coordinates=self.coordinates)
            probe.add_variable("one")
            probe.set_values("one", np.ones(len(probe.entity_points())))
            self._domain_volume = float(probe.integrate("one"))
        return self._domain_volume

    # ---- results ---------------------------------------------------------
    @property
    def mesh(self):
        return self._mesh

    @property
    def time(self) -> float:
        return self._problem.time

    @time.setter
    def time(self, value: float) -> None:
        self._problem.time = value

    def variable_index(self, name: str) -> int:
        return self._problem.variable_index(name)

    def values(self, variable: str) -> np.ndarray:
        """Values of a variable, one per degree of freedom entity.

        For every method except ``"zfvm"`` these are nodal values indexed by
        node.  For ``"zfvm"`` they are cell values followed by boundary face values.
        Use :meth:`entity_points` for the matching coordinates.
        """
        return self._problem.values(variable)

    def set_values(self, variable: str, values) -> None:
        self._problem.set_values(variable, list(np.asarray(values, dtype=float).ravel()))

    def set_element_field(self, name: str, values) -> None:
        """Set a named field with one value per element, which the property objects read
        by name (for instance the accumulated gaseous swelling of
        ``UO2_volumetric_swelling_eigenstrain``).  Setting it again replaces
        the values, and the property objects see the new values at the next solve."""
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != self._mesh.num_elements:
            raise ValueError(f"Element field '{name}': {len(values)} values for {self._mesh.num_elements} elements.")
        self._problem.set_element_field(name, list(values))

    def element_field(self, name: str) -> np.ndarray:
        """The values of a named element field."""
        return np.asarray(self._problem.element_field(name), dtype=float)

    def solution(self) -> np.ndarray:
        """The full solution vector (node-major, variable-minor)."""
        return self._problem.solution()

    def apply_initial_conditions(self) -> None:
        self._problem.apply_initial_conditions()

    def reactions(self, variable: str, boundary: str):
        """Secondary variables at the nodes of a boundary: ``[(node, value), ...]``.

        For each node the value is the integral of the normal flux over the part
        of the boundary that belongs to that node's control domain, which is the
        quantity Reddy denotes :math:`Q_I` (a reaction, a heat flow, a force).

        A node that lies on two boundaries (a corner) carries one reaction that
        covers its whole boundary portion, so it appears in both lists.  When
        summing over several boundaries, collect the nodes first
        (``dict(problem.reactions(...))``) instead of adding the totals, or
        corner nodes are counted twice.
        """
        return self._problem.reactions(variable, boundary)

    def total_reaction(self, variable: str, boundary: str) -> float:
        """Sum of the secondary variables over a boundary."""
        return self._problem.total_reaction(variable, boundary)

    def sample(self, variable: str, points) -> np.ndarray:
        """Interpolate a variable at arbitrary points (NaN outside the mesh)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] < 3:
            points = np.hstack([points, np.zeros((points.shape[0], 3 - points.shape[1]))])
        return np.asarray(self._problem.sample(variable, [list(p) for p in points]))

    def values_at_nodes(self, variable: str, nodes: Iterable[int]) -> np.ndarray:
        values = self.values(variable)
        return np.asarray([values[int(n)] for n in nodes])

    def nodes_where(self, predicate) -> list[int]:
        """Node indices whose coordinates satisfy ``predicate(x, y, z)``."""
        points = self._mesh.points()
        return [i for i, p in enumerate(points) if predicate(p[0], p[1], p[2])]

    def node_at(self, point, tolerance: float = 1e-9) -> int:
        """Index of the node closest to ``point`` (an error if none is within ``tolerance``)."""
        point = np.asarray(list(point) + [0.0] * (3 - len(point)), dtype=float)
        points = np.asarray(self._mesh.points())
        distances = np.linalg.norm(points - point, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            raise ValueError(f"No node within {tolerance} of {point.tolist()}; nearest is at {points[index].tolist()} (distance {distances[index]:.3e}).")
        return index

    def gradient_at_centroids(self, variable: str) -> np.ndarray:
        return np.asarray(self._problem.gradient_at_centroids(variable))

    def property_at_centroids(self, property_name: str) -> np.ndarray:
        return np.asarray(self._problem.property_at_centroids(property_name))

    def kernel_flux_at_centroids(self, kernel_name: str) -> np.ndarray:
        return np.asarray(self._problem.kernel_flux_at_centroids(kernel_name))

    def error_indicator(self, variable: str) -> np.ndarray:
        """One error indicator per element, from gradient recovery.

        The gradient of the computed solution jumps between elements.  A
        smoother gradient is recovered by averaging the element gradients onto
        the nodes, weighted by the share of each element that belongs to the
        node's control domain, and interpolating that nodal field back over the
        element.  The indicator of an element is the square root of the
        integral over it of the squared difference between the two gradients.
        The recovered gradient is the more accurate of the two, so their
        difference measures the error in the computed one.  This is the
        estimator of Zienkiewicz and Zhu (1987).

        It is an indicator, not a bound.  It says which elements carry most of
        the error, which is what :func:`~dualmesh.mark_by_fraction`
        and its relatives need, and it does not certify the size of the error.
        """
        return np.asarray(self._problem.error_indicator(variable))

    def error_norms(self, variable: str, exact, exact_gradient=None, quadrature_points: int = 0) -> tuple[float, float]:
        r"""The error of the computed field against a known exact solution.

        Returns ``(l2, h1_seminorm)``: the :math:`L^2` norm of
        :math:`u_h - u` and the :math:`H^1` seminorm, the :math:`L^2` norm of
        :math:`\nabla u_h - \nabla u`.  These are the norms in which the
        convergence theory of every method in the library is stated, so they
        are what a convergence study should measure.

        ``exact`` is anything a parameter accepts: a number, an expression
        string, a :class:`~dualmesh.ParsedFunction` or a Python callable of
        ``(x, y, z, t)``.  ``exact_gradient`` is a sequence of up to three of
        the same, one per component.  Missing components are taken as zero, and
        without it the seminorm is returned as ``nan``.

        :math:`u_h` is the field the method actually represents: the element
        interpolation of the nodal values for ``dmcdm``, ``fem`` and ``hfvm``,
        and for ``zfvm`` the linear reconstruction
        :math:`U_c + G_c \\cdot (x - x_c)` in every cell, whose gradient is the
        reconstructed cell gradient.  The integrals use a Gauss rule of
        ``quadrature_points`` per direction on every element (zero chooses the
        polynomial order plus two, which over-integrates the leading term of
        the error) and include the coordinate factor.  The sum runs on several
        threads unless one of the functions is a Python callable.
        """
        l2, h1 = self._problem.error_norms(variable, _as_function(exact), None if exact_gradient is None else [_as_function(g) for g in exact_gradient], quadrature_points)
        return float(l2), float(h1)

    def linear_system(self):
        r"""The residual and the Jacobian of the steady problem at the current
        solution, as ``(residual, jacobian)``.

        This is the system one Newton step of :meth:`solve` solves,
        :math:`J\,\delta U = -R`: the prescribed boundary values are written
        into the solution first, and the rows (and columns) of the prescribed
        degrees of freedom are replaced by those of the identity.  ``residual``
        is a NumPy array and ``jacobian`` a SciPy compressed sparse column
        matrix, so the system can be handed to any solver or preconditioner
        that works with SciPy, for instance to compare linear solvers or to
        study the spectrum of a discretisation.  It needs SciPy.  The degree
        of freedom of variable ``v`` on entity ``i`` is ``i * num_variables +
        v``.
        """
        try:
            import scipy.sparse as sparse
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError("Problem.linear_system needs SciPy: pip install scipy") from error
        self._build_physics()
        residual, values, indices, indptr = self._problem._linear_system()
        n = residual.shape[0]
        jacobian = sparse.csc_matrix((values, indices, indptr), shape=(n, n))
        return residual, jacobian

    def boundary_measure(self, boundary) -> float:
        r"""The measure of one side set, or of a list of side sets: the length of
        the boundary in two dimensions and its area in three.  In axisymmetric and
        spherical coordinates the measure includes the coordinate factor, i.e., it
        is the area :math:`\int 2 \pi r \, \mathrm{d}s` of the surface of
        revolution.  It is the area :math:`A` over which the ``total_force`` of a
        ``traction_boundary_condition`` is spread."""
        names = [boundary] if isinstance(boundary, str) else list(boundary)
        return self._problem.boundary_measure(names)

    def integrate(self, variable: str) -> float:
        """The integral of a variable over the domain, with the coordinate
        factor of the problem."""
        return self._problem.integrate(variable)

    def element_integrals(self, variable: str) -> np.ndarray:
        """The integral of a variable over every element (one value per
        element, with the coordinate factor of the problem), e.g., for the
        average of a flux over a fuel assembly."""
        return np.asarray(self._problem.element_integrals(variable), dtype=float)

    def element_volumes(self) -> np.ndarray:
        """The volume of every element, with the coordinate factor of the
        problem (e.g., :math:`2 \\pi r` in axisymmetric coordinates)."""
        return np.asarray(self._problem.element_integrals(""), dtype=float)

    def boundary_flux_integral(self, kernel_name: str, boundary: str) -> float:
        return self._problem.boundary_flux_integral(kernel_name, boundary)

    # ---- output ----------------------------------------------------------
    def write_vtu(self, filename: str, cell_properties: Sequence[str] = (), fields: Sequence[str] = ()) -> None:
        """Write a VTK unstructured grid (readable by ParaView and VisIt).

        ``fields`` are the variables that the file contains (default: all of
        them), and ``cell_properties`` are the properties that it contains as
        cell data.  Each data array of a variable carries its unit in the
        attribute ``units``.  A distributed problem writes one ``.vtu`` file
        per process and a ``.pvtu`` index, which ParaView and VisIt open as one
        data set."""
        if self._distributed is not None:
            base = filename[:-4] if filename.endswith((".vtu", ".pvtu")) else filename
            base = base[:-1] if base.endswith(".") else base
            self._distributed.write_vtu(base, list(cell_properties), list(fields))
            return
        self._problem.write_vtu(filename, list(cell_properties), list(fields))

    # ---- distributed problems ----------------------------------------------
    @property
    def is_distributed(self) -> bool:
        """Whether the problem is split among MPI processes."""
        return self._distributed is not None

    @property
    def rank(self) -> int:
        """This process's rank (0 for a serial problem)."""
        return self._distributed.rank() if self._distributed is not None else 0

    @property
    def num_ranks(self) -> int:
        """The number of processes the problem is split among (1 for a serial
        problem)."""
        return self._distributed.num_ranks() if self._distributed is not None else 1

    @property
    def num_owned_dofs(self) -> int:
        """Degrees of freedom this process owns (all of them in a serial
        problem)."""
        return self._distributed.num_owned_dofs() if self._distributed is not None else self.num_active_dofs()

    @property
    def num_global_dofs(self) -> int:
        """Degrees of freedom of the whole problem."""
        return self._distributed.num_global_dofs() if self._distributed is not None else self.num_active_dofs()

    def gathered_values(self, variable: str) -> np.ndarray:
        """Values of a variable at every node of the whole mesh, on every
        process.  A distributed problem allocates one vector of the global
        size per process, so this suits tests and small problems, and a large
        run writes its result with :meth:`write_vtu`.  For a serial problem it
        equals :meth:`values`."""
        if self._distributed is not None:
            return np.asarray(self._distributed.gathered_values(variable))
        return self.values(variable)

    def partition_summary(self) -> str:
        """One line describing the partition of a distributed problem."""
        return self._distributed.summary() if self._distributed is not None else "serial problem (one process)"

    def write_mesh_file(self, filename: str, file_format: str | None = None) -> None:
        """Write the mesh and all nodal fields through meshio (Exodus, VTU, ...)."""
        from .meshing import write_mesh

        fields = {}
        for index in range(self._problem.num_variables):
            name = self._problem.variable_name(index)
            fields[name] = self.values(name)
        write_mesh(self._mesh, filename, file_format=file_format, **fields)

    def write_csv(self, filename: str, variables: Sequence[str] = ()) -> None:
        """Write the values of the variables as comma-separated values: one
        row for each degree of freedom (a node, or a cell and a boundary face
        of the cell-centred method), with its coordinates.  The header names
        every column with its unit, for example ``temperature (K)``.
        ``variables`` selects the variables (default: all of them)."""
        from .tables import column_header

        names = list(variables) or [self._problem.variable_name(i) for i in range(self._problem.num_variables)]
        points = np.asarray(self.entity_points())
        dimension = self._mesh.dimension
        columns = [points[:, i] for i in range(dimension)]
        columns += [self.values(name) for name in names]
        header = ",".join([column_header(axis, "m") for axis in ("x", "y", "z")[:dimension]] + [column_header(name, self.variable_unit(name)) for name in names])
        np.savetxt(filename, np.column_stack(columns), delimiter=",", header=header, comments="")

    def summary(self, parameters: bool = True) -> str:
        """A report of the problem: method, mesh, variables and objects, with
        every parameter of every object and its value, marked ``(default)``
        when it was not set.  See :mod:`dualmesh.console`."""
        from .console import problem_report

        self._build_physics()
        return problem_report(self, parameters=parameters)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<dualmesh.Problem method={self.method} {self._mesh.num_nodes} nodes>"
