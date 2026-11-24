# SPDX-License-Identifier: LGPL-2.1-or-later
"""The :class:`Problem` class: variables, objects, solvers, and results."""

from __future__ import annotations

import difflib
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from . import _core
from ._core import InputError
from .console import report_level
from .output import Output
from .tables import ResultTables

SolveResult = _core.SolveResult


def _solve_result_tables(self) -> dict:
    """``newton``: one row per Newton iteration."""
    from .tables import Table

    return {
        "newton": Table(
            ["continuation_step", "data_fraction", "iteration", "residual_norm", "step_norm"],
            None,
            [
                [r.continuation_step, r.data_fraction, r.iteration, r.residual_norm, r.step_norm]
                for r in self.history
            ],
            title="Newton iterations",
        )
    }


def _solve_result_to_dict(self) -> dict:
    """Whether the solve converged, its iteration counts and the history of
    its Newton iterations."""
    history = [
        dict(
            continuation_step=r.continuation_step,
            data_fraction=r.data_fraction,
            iteration=r.iteration,
            residual_norm=r.residual_norm,
            step_norm=r.step_norm,
        )
        for r in self.history
    ]
    return {
        "study": "solve",
        "converged": bool(self.converged),
        "total_iterations": int(self.total_iterations),
        "linear_iterations": int(self.linear_iterations),
        "time_steps": int(self.time_steps),
        "rejected_steps": int(self.rejected_steps),
        "history": history,
    }


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


#: The time schemes of solve_transient, by the weight theta of their steady terms at the new time.
TIME_SCHEMES = {"backward_Euler": 1.0, "Crank-Nicolson": 0.5, "forward_Euler": 0.0}


def _theta_of(time_scheme: str) -> float:
    """The theta of a time scheme; its name may be written in lower case."""
    for name, theta in TIME_SCHEMES.items():
        if str(time_scheme).lower() == name.lower():
            return theta
    raise InputError(
        f"solve_transient: unknown time_scheme '{time_scheme}'. Use {', '.join(TIME_SCHEMES)}."
    )


def _object_name(obj) -> str:
    """The internal name of an object that an add method returned (or the name itself)."""
    if isinstance(obj, str):
        return obj
    name = obj.name
    return name() if callable(name) else name


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
        "relaxation",
        "line_search",
        "max_line_search_steps",
        "linear_tolerance",
        "petsc_options",
        "error_on_divergence",
    }
    for key, value in kwargs.items():
        if key == "verbose":
            raise TypeError(
                "Unknown solver option 'verbose'. The parameter report sets what a solve prints: report='full' (the default), 'summary' or 'none'."
            )
        if key == "_progress":
            options.verbose = bool(value)
            continue
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise TypeError(
                f"Unknown solver option '{key}'.{hint} Known options: {', '.join(sorted(known))}."
            )
        if key == "petsc_options":
            value = petsc_option_string(value)
        if key == "nonlinear_solver" and isinstance(value, str):
            value = value.lower()
        setattr(options, key, value)
    return options


#: The methods, written as the abbreviations they are (CODE_AND_DOCS_RULES 4.1); the lower-case
#: spelling is accepted.
METHODS = {"fem": "FEM", "dmcdm": "DMCDM"}


# ---------------------------------------------------------------------------
# Destruction of distributed problems at collective points
# ---------------------------------------------------------------------------
# A distributed problem holds MPI, PETSc and gslib objects that every process must destroy
# together.  Python's cycle collector frees an unreachable problem whenever each process happens
# to run it, so one process could block in a destructor while another waits in a solve.  When a
# distributed problem is collected, its C++ object is kept here instead, and the processes
# destroy the kept objects together, in the order of their creation, at the collective points:
# the creation of a problem and the start of a solve.  Before that, every process collects its
# cycles, so that the processes of one script hold the same unreachable problems at that point.
_KEPT_FOR_COLLECTIVE_DESTRUCTION: list = []
_CREATED = [0]


def _keep_for_collective_destruction(serial: int, distributed) -> None:
    _KEPT_FOR_COLLECTIVE_DESTRUCTION.append((serial, distributed))


def _destroy_collectively() -> None:
    """Destroy the kept distributed problems, on every process at the same point."""
    if _core.mpi_size() <= 1:
        return
    import gc

    gc.collect()
    _KEPT_FOR_COLLECTIVE_DESTRUCTION.sort(key=lambda kept: kept[0])
    while _KEPT_FOR_COLLECTIVE_DESTRUCTION:
        _KEPT_FOR_COLLECTIVE_DESTRUCTION.pop(0)


def _destroy_at_exit() -> None:
    # Registered when the module is imported, before the first finalizer of weakref registers its
    # own exit hook, so that it runs after the finalizers have kept the last problems.
    _destroy_collectively()


import atexit as _atexit  # noqa: E402

_atexit.register(_destroy_at_exit)


def _changes_model(method):
    """Mark the solution as no longer that of the model: a method that adds
    to the problem or changes a parameter."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # A refused call changes nothing, so only a call that returns marks the solution.
        result = method(self, *args, **kwargs)
        self._steady_solution = False
        return result

    return wrapper


class Problem:
    """A boundary value problem discretized on a primal mesh and its dual mesh.

    Parameters
    ----------
    mesh:
        The primal mesh of finite elements.
    method:
        The discretization: ``"FEM"``, the Galerkin finite element method, for
        solids, or ``"DMCDM"``, the dual mesh control domain method, which
        conserves mass and heat in every control volume, for fluids.  The
        default, ``"automatic"``, takes DMCDM when the problem has a
        ``fluid_dynamics`` of the pressure formulation and FEM otherwise, and
        is decided when the physics are built.  The rest of the problem
        definition is the same for both.
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
        distributed linear solvers (see :meth:`solve`).  Both methods run
        distributed.
    partitioner:
        The partitioner of PETSc that splits the elements of a distributed problem among the processes: ``"automatic"`` (the default: PT-Scotch, else ParMETIS), ``"ptscotch"``, ``"parmetis"`` or ``"simple"``.
        See :func:`dualmesh.partition_mesh`.
    gather_scatter:
        The library that exchanges, between the processes of a distributed problem, the values at the nodes that several of them hold.
        ``"petsc"`` (the default) is PETSc's star forests, which also carry PETSc's own vectors and matrices.
        ``"gslib"`` is the gather-scatter library of Nek5000, which at setup times pairwise exchanges, a crystal router and an all-reduce and keeps the fastest for the mesh at hand (:func:`dualmesh.have_gslib`).
        Both give the same answer, and the time per exchange may differ on many processes.
    """

    def __init__(
        self,
        mesh,
        method: str = "automatic",
        coordinates: str = "cartesian",
        distributed: bool | None = None,
        partitioner: str = "automatic",
        gather_scatter: str = "petsc",
    ):
        if distributed is None:
            distributed = _core.mpi_size() > 1
        # The automatic method starts as FEM and becomes DMCDM if a flow is built (_build_physics).
        automatic = str(method).lower() == "automatic"
        self._automatic_method = "open" if automatic else "given"
        given, method = method, "FEM" if automatic else METHODS.get(str(method).lower())
        if method is None:
            raise _core.InputError(
                f"Unknown method '{given}'. Use 'FEM' (solids), 'DMCDM' (fluids) or 'automatic'."
            )
        self._distributed = None
        _destroy_collectively()
        if distributed:
            options = _core.DistributedOptions()
            options.partitioner = partitioner
            options.gather_scatter = gather_scatter
            self._distributed = _core.DistributedProblem(mesh, method.lower(), coordinates, options)
            self._problem = self._distributed.local()
            import weakref

            _CREATED[0] += 1
            weakref.finalize(self, _keep_for_collective_destruction, _CREATED[0], self._distributed)
            self._global_mesh = mesh
            mesh = self._problem.mesh
        else:
            self._problem = _core.Problem(mesh, method.lower(), coordinates)
        self._mesh = mesh
        self._objects: list = []  # keeps Python-defined objects alive
        self._init_postprocessing()
        self._physics: dict = {}
        # True while the solution is a converged steady solution of the model as it stands.
        self._steady_solution = False
        self._couplings: dict = {}
        # The velocity that a run continued from another mesh starts from (set_gathered_velocities).
        self._continued_velocity: dict = {}
        self.method = method
        self.coordinates = coordinates

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
        self._extra_step_callback = None
        self._domain_volume = None

    def node_coordinates(self) -> numpy.ndarray:
        """The coordinates of the nodes, one row each, in the order of
        :meth:`values`."""
        return numpy.asarray(self._mesh.points())

    def gathered_node_coordinates(self) -> numpy.ndarray:
        """The coordinates of the nodes of the whole mesh, in the order of
        :meth:`gathered_values`; in a serial problem they equal :meth:`node_coordinates`."""
        if self._distributed is not None:
            return numpy.asarray(self._global_mesh.points())
        return self.node_coordinates()

    # ---- definition ------------------------------------------------------
    def _set_initial_velocity(self, time: float) -> None:
        """The initial velocity of the second-order variables: the one given by
        set_gathered_velocities, else that of the initial_condition of their physics (at rest
        without one).  A run that continues another of the same problem keeps the velocity it
        reached."""
        names = list(self._problem.variable_names())
        points = self.node_coordinates()
        velocity = numpy.zeros(len(points) * len(names))
        for physics in self._physics.values():
            for variable in physics.variable_names():
                index = names.index(variable)
                if variable in self._continued_velocity:
                    whole = self._continued_velocity[variable]
                    velocity[index :: len(names)] = (
                        self._distributed.local_node_values(list(whole), 1)
                        if self._distributed is not None
                        else whole
                    )
                    continue
                value = physics._initial_velocity(variable)
                if value is None:
                    continue
                f = _core.ParsedFunction(str(value))
                for node, p in enumerate(points):
                    velocity[node * len(names) + index] = f(p[0], p[1], p[2], time)
        self._problem.set_initial_velocity(list(velocity))

    def gathered_velocities(self) -> dict:
        """The velocity that the last transient run reached, for each variable of a second-order
        equation, at every node of the whole mesh, on every process: zero before a run, and an
        empty dict for a problem without inertia.  Collective in a distributed problem."""
        self._build_physics()
        second_order = list(self._problem.second_order_variables())
        if not second_order:
            return {}
        names = list(self._problem.variable_names())
        local = numpy.asarray(self._problem.velocity(), dtype=float)
        if local.size == 0:
            local = numpy.zeros(len(self.node_coordinates()) * len(names))
        whole = (
            numpy.asarray(self._distributed.gathered_node_values(list(local), len(names)))
            if self._distributed is not None
            else local
        )
        return {name: whole[names.index(name) :: len(names)].copy() for name in second_order}

    def set_gathered_velocities(self, velocities: dict) -> None:
        """Start the next transient run with these velocities, given at every node of the whole
        mesh for variables of second-order equations, in place of those of the initial conditions:
        for a run that goes on from the state that another reached, such as after a change of the
        mesh."""
        self._continued_velocity = {
            name: numpy.asarray(values, dtype=float).ravel() for name, values in velocities.items()
        }

    @_changes_model
    def add_variable(
        self,
        name: str,
        block: Sequence[str] = (),
        initial_condition=None,
        order: str = "mesh",
        unit: str = "",
    ) -> int:
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

    @_changes_model
    def add_function(self, name: str, function) -> None:
        """Register a named function of ``(x, y, z, t)`` (or a constant)."""
        self._problem.add_function(name, function)

    def _add(self, adder, object_or_type, kwargs):
        if isinstance(object_or_type, str):
            # A matrix (a list of rows of numbers) is passed row by row.
            kwargs = {
                k: [float(x) for row in v for x in row]
                if isinstance(v, (list, tuple))
                and v
                and all(isinstance(r, (list, tuple, numpy.ndarray)) for r in v)
                else v
                for k, v in kwargs.items()
            }
            return self._problem.add_object(object_or_type, **kwargs)
        if kwargs:
            raise TypeError("Extra parameters are not accepted when adding an object instance.")
        self._objects.append(object_or_type)
        adder(object_or_type)
        return object_or_type

    @_changes_model
    def add_kernel(self, kernel, **parameters):
        """Add a term to an equation: a kernel of the object reference by its
        type name, or a kernel written in Python::

            problem.add_kernel("diffusion", variable="u", diffusivity=2.0)
        """
        return self._add(self._problem.add_kernel, kernel, parameters)

    @_changes_model
    def add_boundary_condition(self, condition, boundary=None, **parameters):
        """Add a boundary condition (essential or natural) on the side sets or
        node sets named by ``boundary``::

            problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
            problem.add_boundary_condition("Dirichlet", ["left", "right"], variable=["u", "v"])

        A list of variables adds the condition to each of them.  A periodic
        condition names its two boundaries with ``primary`` and ``secondary``,
        and ``point_Dirichlet`` acts at a ``point``, so neither takes
        ``boundary``."""
        if boundary is not None:
            parameters["boundary"] = [boundary] if isinstance(boundary, str) else list(boundary)
            known = set(self._mesh.sideset_names()) | set(self._mesh.nodeset_names())
            for name in parameters["boundary"]:
                if name not in known:
                    close = difflib.get_close_matches(name, sorted(known), n=1)
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise ValueError(
                        f"{condition}: the mesh has no boundary '{name}'.{hint} Its boundaries are: {', '.join(sorted(known)) or 'none'}."
                    )
        if isinstance(condition, str):
            category = _core.object_category(condition)
            if category not in ("boundary_condition", "nodal_boundary_condition", "constraint"):
                raise ValueError(f"'{condition}' is a {category}, not a boundary condition.")
            return self._problem.add_object(condition, **parameters)
        if isinstance(condition, _core.NodalBC):
            return self._add(self._problem.add_nodal_bc, condition, parameters)
        return self._add(self._problem.add_integrated_bc, condition, parameters)

    @_changes_model
    def add_property(self, property_type, **parameters):
        """Add a property object: an object that computes named properties at
        the integration points of its blocks (e.g., a thermal conductivity,
        a stress or the cross sections of a region), which the physics, the
        kernels and the expressions read by name::

            problem.add_property(
                "constant", block=["bar"], properties={"thermal_conductivity": 400.0}
            )
        """
        return self._add(self._problem.add_property, property_type, parameters)

    @_changes_model
    def add_point_source(self, **parameters):
        """Add a concentrated nodal source (point force or point heat source).
        ``points`` gives one coordinate per dimension of the mesh for each point,
        as one list per point or as one flat list."""
        if "points" in parameters:
            parameters["points"] = [float(c) for c in numpy.ravel(parameters["points"])]
        return self._add(self._problem.add_nodal_load, "point_source", parameters)

    @_changes_model
    def add_physics(self, physics: str, **parameters):
        """Add a set of equations named in physical terms (see
        :mod:`dualmesh.physics` and ``dualmesh list --category physics``).

        The physics creates its variables now and generates its kernels and
        materials when the problem is initialised.  It returns the physics,
        whose :meth:`~dualmesh.physics.Physics.add_boundary_condition` and
        :meth:`~dualmesh.physics.Physics.add_kernel` add conditions and terms
        to its equations, and which a coupling takes to name it::

            heat = problem.add_physics(
                "heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6
            )
            heat.add_boundary_condition("Dirichlet", "left", value=40.0)
        """
        from . import physics as _physics

        instance = _physics.create(
            physics, _physics.unique_name(physics, self._physics), parameters
        )
        if not isinstance(instance, _physics.Physics):
            raise _physics.InputError(f"'{physics}' is a coupling. Add it with add_coupling.")
        instance._attach(self)
        self._physics[instance.name] = instance
        return instance

    @_changes_model
    def add_coupling(self, coupling: str, **parameters):
        """Couple two physics of the problem, e.g., ``thermal_expansion`` or
        ``nonisothermal_flow`` (``dualmesh list --category coupling``)::

            heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0)
            solid = problem.add_physics(
                "solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3
            )
            problem.add_coupling(
                "thermal_expansion",
                heat_transfer=heat,
                solid_mechanics=solid,
                linear_expansion_coefficient=1.2e-5,
                stress_free_temperature=293.15,
            )

        A coupling takes each of its physics as the object that
        :meth:`add_physics` returned.  A physics may be left out when the
        problem has only one physics of its type.
        """
        from . import physics as _physics

        name = _physics.unique_name(coupling, self._couplings)
        instance = _physics.create(coupling, name, parameters)
        if not isinstance(instance, _physics.Coupling):
            raise _physics.InputError(f"'{coupling}' is a physics. Add it with add_physics.")
        if any(p._built for p in self._physics.values()):
            raise _physics.InputError(
                f"{coupling}: add the couplings before the problem is solved or initialised."
            )
        instance._apply(self)
        self._couplings[name] = instance
        return instance

    def _build_physics(self) -> None:
        """Generate the objects of the physics and couplings not built yet."""
        new = [p for p in self._physics.values() if not p._built]
        if not new:
            return
        # The automatic method: DMCDM for a flow of the pressure formulation, FEM otherwise,
        # chosen when the physics are first built.
        flow = any(p.type_name == "fluid_dynamics" and p.formulation == "equal_order" for p in new)
        if self._automatic_method == "open":
            if flow:
                self._problem.set_method("dmcdm")
                self.method = "DMCDM"
            self._automatic_method = "chosen"
        elif self._automatic_method == "chosen" and flow and self.method == "FEM":
            raise _core.InputError(
                "The method was chosen as FEM when the problem was first solved, before this flow "
                "was added. Give method='DMCDM' to dualmesh.Problem for the flow."
            )
        for physics in new:
            physics._build(self)
            physics._built = True
        for coupling in self._couplings.values():
            if not getattr(coupling, "_built", False):
                coupling._build(self)
                coupling._built = True
        # The advection scheme of a heat transfer physics applies to the heat a flow carries, which
        # only a nonisothermal_flow coupling adds.
        carried = {
            id(c.heat_transfer)
            for c in self._couplings.values()
            if getattr(c, "type_name", "") == "nonisothermal_flow"
        }
        for physics in new:
            if (
                getattr(physics, "type_name", "") == "heat_transfer"
                and physics.advection_scheme is not None
                and id(physics) not in carried
            ):
                raise InputError(
                    f"heat_transfer: advection_scheme '{physics.advection_scheme}' discretises the heat that a flow carries, and no nonisothermal_flow coupling makes a flow carry this heat."
                )

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

        When Newton's method does not converge on the problem itself, the
        solve takes it again by continuation: it ramps the data that drive
        the problem (the sources, the body forces and buoyancy, the prescribed
        fluxes, tractions and loads) up from zero, and the prescribed values
        from those of the starting state, in steps that it halves after a
        failure and doubles after a success.  A flow at a high Reynolds
        number, a strongly nonlinear material or a large deflection needs no
        setting for it.  ``result.history`` records each step.

        Keyword arguments are solver options: ``nonlinear_solver`` (``"Newton"``,
        ``"Picard"``, or ``"linear"``), ``max_iterations``, ``relative_tolerance``
        (each equation against the size of its own terms), ``relaxation`` (the
        acceleration parameter of direct iteration), ``line_search``
        (``"automatic"``, the default, which is backtracking when the limited
        advection scheme is used, whose limiter has kinks on which the full
        Newton step can cycle, and none otherwise; ``"none"``; or
        ``"backtracking"``, which halves a Newton step until the scaled
        residual falls) and ``max_line_search_steps``
        (default 10), ``linear_tolerance`` (the relative tolerance of an
        iterative linear solve), ``petsc_options`` (PETSc's options, as a
        string or a dictionary) and ``error_on_divergence``.
        :doc:`/user_guide/solving` explains every option and its default.

        SNES, the nonlinear solver of PETSc, runs the iteration, and PETSc
        solves its linear systems, the same way in a serial problem and in a
        distributed one (see the parameter ``distributed`` of the class).
        Without ``petsc_options``, the solver factorises the system directly
        where that is cheap, which is always in one dimension, up to
        :math:`10^5` unknowns in two and up to 4000 in three.  A larger flow
        takes flexible GMRES with the least squares commutator, a solid GMRES
        with smoothed aggregation (GAMG) and its rigid body modes, a solid
        coupled to other physics the split by physics, and any other system
        GMRES with hypre BoomerAMG, each at the linear tolerance of Eisenstat
        and Walker, which follows the decrease of the residual.  When the
        iteration does not converge, the direct factorisation follows.
        Options that start with ``-snes_`` go to SNES, and the others replace
        this choice with PETSc's own options of the linear solver, for
        example ``petsc_options="-ksp_type preonly -pc_type lu"``.  Two options of
        dualmesh-multiphysics choose a split that PETSc cannot name:
        ``-dualmesh_schur lsc``, ``pcd`` or ``pressure_mass`` splits a flow
        with that approximation of the Schur complement, and
        ``-dualmesh_split physics`` splits the system by physics.
        :doc:`/theory/parallel` gives the method and the iteration counts.
        """
        level = report_level(report, "Problem.solve")
        output = self._check_output(output, steady=True)
        self._build_physics()
        self._report_start(level, "steady solve")
        start = time.perf_counter()
        options["_progress"] = level == "full"
        if self._distributed is not None:
            _destroy_collectively()
            result = self._distributed.solve_steady(_solver_options(**options))
        else:
            result = self._problem.solve_steady(_solver_options(**options))
        self._steady_solution = bool(result.converged)
        if self._postprocessors.postprocessors:
            self._postprocessors.evaluate(self, self.time)
        files = self._write_fields(output) if output is not None else []
        self._report_end(level, result, time.perf_counter() - start, files)
        return result

    # ---- adjoint sensitivities ----------------------------------------------
    @_changes_model
    def add_parameter(self, name: str, value: float) -> None:
        """Register a named parameter that :meth:`sensitivities` differentiates
        with respect to: a settable constant, read by its name wherever a
        coefficient, a source or a boundary value accepts a function.  Register
        the parameters before the objects that read them."""
        self._problem.add_parameter(name, float(value))

    @_changes_model
    def set_parameter(self, name: str, value: float) -> None:
        """Assign a parameter registered by :meth:`add_parameter`; the objects
        that read it see the new value at the next solve."""
        self._problem.set_parameter(name, float(value))

    def boundary_integral(self, variable: str, boundary: str) -> float:
        """The integral of a variable over one side set, with the coordinate
        factor: the value of the boundary objective of :meth:`sensitivities`."""
        return self._problem.objective_value(variable, boundary)

    def sensitivities(
        self, objective, parameters, report: str = "full", **options
    ) -> SensitivityResult:
        """The derivative of an objective of the solution with respect to named
        parameters, by one adjoint solve.

        ``objective`` is the name of a variable, for its integral over the whole
        domain, or a dict ``{"variable": ..., "boundary": ...}`` for its integral
        over one side set.  ``parameters`` lists the names given to
        :meth:`add_parameter`.  The problem must have been solved first: the
        sensitivities are those of the converged solution.  Keyword arguments
        are the linear solver options of :meth:`solve`, for the adjoint solve.

        With ``R(U, p) = 0`` the solved residual, ``g`` the objective and ``J``
        the exact Jacobian at the solution, the adjoint ``lambda`` solves
        ``J^T lambda = (dg/dU)^T`` and every sensitivity follows from
        ``dg/dp = dg/dU dU/dp - lambda (dR/dp + J dU/dp)`` on the free degrees
        of freedom, where ``dU/dp`` is nonzero only where a prescribed value
        depends on the parameter.  ``dR/dp`` is a central difference of the
        residual at the converged solution (one assembly per parameter, no
        solve), so the study costs one adjoint solve however many unknowns the
        problem has, and one assembly per parameter.
        """
        level = report_level(report, "Problem.sensitivities")
        if self._distributed is not None:
            raise InputError(
                "sensitivities: a distributed problem is not differentiated yet; run the study on one process."
            )
        if not self._steady_solution:
            raise InputError(
                "sensitivities: the problem holds no converged steady solution of the model as it stands. Call solve() first, after the last change of the model or of a parameter; a transient or eigenvalue study is not differentiated."
            )
        self._build_physics()
        if isinstance(objective, str):
            variable, boundary = objective, ""
        elif isinstance(objective, dict) and isinstance(objective.get("variable"), str):
            variable, boundary = objective["variable"], objective.get("boundary", "")
        else:
            raise InputError(
                "sensitivities: the objective is the name of a variable (its integral over the "
                f"domain) or a dict {{'variable': ..., 'boundary': ...}}, not {objective!r}."
            )
        if not isinstance(parameters, (list, tuple)) or not parameters:
            raise InputError(
                "sensitivities: parameters is the list of the names given to add_parameter, "
                f"not {parameters!r}."
            )
        start = time.perf_counter()
        values = self._problem.sensitivities(
            variable, boundary, list(parameters), _solver_options(**options)
        )
        result = SensitivityResult(objective=(boundary, variable), values=dict(values))
        if level != "none":
            from .console import header

            if level == "full":
                print(header("sensitivity study"))
                print(self.summary(), flush=True)
            print(result.summary())
            print(f"  ({time.perf_counter() - start:.2f} s)")
        return result

    # ---- reports and files of the solves ------------------------------------
    def _report_start(self, level: str, study: str) -> None:
        if level == "full":
            from .console import header

            print(header(study))
            print(self.summary(), flush=True)
        elif level == "summary":
            print(
                f"dualmesh {study}: {self.num_global_dofs} unknowns, method {self.method}",
                flush=True,
            )

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
            raise InputError(
                f"The output of a solve is a dualmesh.Output group, not {type(output).__name__}."
            )
        if steady and (output.times is not None or output.interval is not None):
            raise InputError(
                "Output: a steady solve writes the fields once, at the end. Remove times and interval."
            )
        if output.fields is not None:
            names = [self._problem.variable_name(i) for i in range(self._problem.num_variables)]
            for name in output.fields:
                if name not in names:
                    close = difflib.get_close_matches(name, names, n=1)
                    hint = f" Did you mean '{close[0]}'?" if close else ""
                    raise InputError(
                        f"Output: the field '{name}' is not a variable.{hint} The variables are: {', '.join(names)}."
                    )
        if self._distributed is not None and "csv" in output.formats:
            raise InputError(
                "Output: a distributed problem writes the vtu format only. Remove csv from formats."
            )
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
            files.append(
                path[:-4] + ".pvtu"
                if self._distributed is not None and self.num_ranks > 1
                else path
            )
        if "csv" in output.formats:
            path = output.path(suffix + ".csv")
            self.write_csv(path, variables=fields)
            files.append(path)
        return files

    @_changes_model
    def solve_transient(
        self,
        end_time: float,
        time_step: float,
        start_time: float = 0.0,
        time_scheme: str | None = None,
        spectral_radius: float | None = None,
        output=None,
        report: str = "full",
        time_stepper: str = "fixed",
        courant_number: float = 0.5,
        min_time_step: float = 0.0,
        max_time_step: float = 0.0,
        growth_factor: float = 2.0,
        cutback_factor: float = 0.5,
        error_tolerance: float = 1.0e-3,
        optimal_iterations: int = 4,
        iteration_window: int = 2,
        max_rejected_steps: int = 10,
        max_steps: int = 0,
        restart: str | None = None,
        **options,
    ) -> SolveResult:
        r"""March the solution forward in time.

        The time schemes are theta methods, which weight the steady part of
        the residual between the old state and the new one,

        .. math::

            R_{\text{time}}(U^{n+1})
            + \theta R_{\text{steady}}(U^{n+1})
            + (1 - \theta) R_{\text{steady}}(U^{n}) = 0,

        with :math:`\theta = 1` for ``"backward_Euler"`` (the default), which is
        unconditionally stable and first-order accurate, :math:`\theta = 1/2`
        for ``"Crank-Nicolson"``, the midpoint rule, which is unconditionally
        stable and second-order accurate and can oscillate after a sharp
        transient, and :math:`\theta = 0` for ``"forward_Euler"``, which is
        explicit in the steady terms and stable only below a critical step.

        A problem without a time derivative (no capacity or mass term) is
        marched quasi-statically: each step solves the steady problem at its
        time, which is what loads that change in time or a material history
        such as creep need.

        Parameters
        ----------
        end_time, time_step, start_time:
            The interval to cover and the step to take.  With an adaptive
            stepper ``time_step`` is the first step.
        time_scheme:
            ``"backward_Euler"`` (the default, stable for every step),
            ``"Crank-Nicolson"`` or ``"forward_Euler"``.  A problem with
            inertia (a ``density`` of ``solid_mechanics``, or an ``m`` of
            ``general_form_PDE``) takes ``"generalized_alpha"`` (the default
            there: the method of Chung and Hulbert, second order, with damping
            of the modes the mesh cannot resolve) or ``"Newmark"`` (the
            average acceleration rule: second order, without numerical
            damping, energy conserved).  Its first-order equations take the
            theta method with :math:`\theta = 1 - \alpha_f`.
        spectral_radius:
            The damping of ``"generalized_alpha"``: the spectral radius
            :math:`\rho_\infty` at infinite frequency, from 0 (the highest
            frequency damped out in one step) to 1 (no damping).  Default 0.75,
            COMSOL's default.
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
            truncation error of the variables with a time derivative.  A step
            of a theta method is compared with the extrapolation of the last
            accepted states, at one nonlinear solve per step.  The first steps,
            an equation with a second time derivative and the projection
            integration of a flow advance each interval twice instead, once
            with one step and once with two half steps, at three solves per
            step, and keep the answer of the half steps.  A step whose relative
            error exceeds ``error_tolerance`` is discarded and retried with a
            smaller step, and an accepted step is followed by the largest step
            the estimate allows.  Use this when accuracy in time is what
            matters.

            ``"iteration"`` chooses the step from how hard the nonlinear solver
            worked.  A step that converged in fewer than
            ``optimal_iterations - iteration_window`` iterations is followed by
            a larger one, a step that needed more than
            ``optimal_iterations + iteration_window`` by a smaller one, and a
            step that failed to converge is discarded and retried.  It costs
            nothing beyond the solve and suits a problem whose difficulty lies
            in the nonlinearity.

            ``"cfl"`` keeps the Courant number of a flow advanced by the
            projection time integration (``time_integration="projection"`` of
            ``fluid_dynamics``) at ``courant_number``: each step is the one
            that gives that Courant number at the velocity of the step before,
            grown by at most ``growth_factor``, as Nek5000 and nekRS choose their
            steps.
        courant_number:
            The Courant number the ``"cfl"`` stepper aims for,
            :math:`\Delta t \max_e |\mathbf{u}| / h_e` over the nodes of every
            element.  Default 0.5, the value Nek5000 and nekRS use with BDF3 and
            EXT3.
        min_time_step, max_time_step:
            Bounds on the step.  A run that has to go below ``min_time_step``
            is reported as a failure.  Zero means ``time_step`` divided by one
            million, and the whole interval, respectively.
        growth_factor, cutback_factor:
            The most the step may grow between accepted steps, and the factor
            applied after a rejected one.
        error_tolerance:
            The target for the local error of one step, each variable
            against its range.
        optimal_iterations, iteration_window:
            The iteration count the ``"iteration"`` stepper aims for, and the
            half-width of the band around it inside which the step is left
            alone.
        max_rejected_steps:
            How many times in a row a step may be rejected before the run is
            declared a failure.
        max_steps:
            Stop after this many accepted steps, at the time they reach, with
            ``result.next_time_step`` the step the stepper chose to take next.
            Default 0: no limit.
        restart:
            A checkpoint (the ``.chk`` file of an :class:`~dualmesh.Output` group with ``checkpoint_interval``) to continue from: the run starts at its time, with its step size, its solution, the history of its material and the history of its post-processors, on any number of processes.
            ``start_time`` stays the start of the original run, so that the output times and the file numbers continue those of the original run.
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
        self._build_physics()
        dynamic = bool(self._problem.second_order_variables())
        if time_scheme is None:
            time_scheme = "generalized_alpha" if dynamic else "backward_Euler"
        if dynamic:
            self._set_initial_velocity(start_time)
            if time_scheme not in ("generalized_alpha", "Newmark"):
                raise InputError(
                    f"solve_transient: a problem with inertia takes the time_scheme 'generalized_alpha' (the default) or 'Newmark', not '{time_scheme}'."
                )
            transient.second_order_scheme = time_scheme
        else:
            transient.theta = _theta_of(time_scheme)
        if spectral_radius is not None:
            if time_scheme != "generalized_alpha":
                raise InputError(
                    "solve_transient: spectral_radius is the damping of the time_scheme 'generalized_alpha'."
                )
            transient.spectral_radius = float(spectral_radius)
        transient.time_stepper = time_stepper
        transient.courant_number = courant_number
        transient.dt_min = min_time_step
        transient.dt_max = max_time_step
        transient.growth_factor = growth_factor
        transient.cutback_factor = cutback_factor
        transient.error_tolerance = error_tolerance
        transient.optimal_iterations = optimal_iterations
        transient.iteration_window = iteration_window
        transient.max_rejected_steps = max_rejected_steps
        transient.max_steps = int(max_steps)
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
                csv_output = Output(
                    directory=output.directory,
                    file_base=output.file_base,
                    fields=output.fields,
                    formats=("csv",),
                )
                self._problem.set_output_callback(
                    lambda index: files.extend(self._write_fields(csv_output, index))
                )
        # A restart continues the run of its checkpoint: its time, its state and the history of its post-processors.
        first_step = 0
        if restart is not None:
            if not _core.have_checkpoints():
                raise InputError(
                    "A restart reads a checkpoint, which needs a build with PETSc (MPI, or -DDUALMESH_ENABLE_PETSC=ON)."
                )
            restart = str(restart)
            _, _, first_step = _core.checkpoint_time(restart, self._distributed is not None)
            transient.restart_file = restart
            history = restart + ".postprocessors.json"
            if self._postprocessors.postprocessors and os.path.exists(history):
                self._postprocessors.load_json(history)
        checkpoint = None
        if output is not None and output.checkpoint_interval:
            if not _core.have_checkpoints():
                raise InputError(
                    "Output: checkpoints need a build with PETSc (MPI, or -DDUALMESH_ENABLE_PETSC=ON)."
                )
            checkpoint = output.path(".chk")
            transient.checkpoint_file = checkpoint
            transient.checkpoint_interval = int(output.checkpoint_interval)
        self._report_start(level, "transient solve")
        start = time.perf_counter()
        if self._postprocessors.postprocessors and restart is None:
            self.time = start_time
            self._postprocessors.evaluate(self, start_time)
        if checkpoint is not None and self._postprocessors.postprocessors:
            # The history of the post-processors goes beside each checkpoint, at the same steps.
            steps = [first_step]
            interval = int(output.checkpoint_interval)
            tolerance = 1e-12 * abs(end_time - start_time)

            def save_history(t, core_problem):
                steps[0] += 1
                if (steps[0] % interval == 0 or t >= end_time - tolerance) and self.rank == 0:
                    self._postprocessors.write_json(checkpoint + ".postprocessors.json")

            self._extra_step_callback = save_history
            self._install_step_callback()
        options["_progress"] = level == "full"
        try:
            if self._distributed is not None:
                _destroy_collectively()
                result = self._distributed.solve_transient(transient, _solver_options(**options))
            else:
                result = self._problem.solve_transient(transient, _solver_options(**options))
        finally:
            if csv_output is not None:
                self._problem.set_output_callback(None)
            if self._extra_step_callback is not None:
                self._extra_step_callback = None
                self._install_step_callback()
        if output is not None and "vtu" in output.formats:
            extension = "pvtu" if self._distributed is not None and self.num_ranks > 1 else "vtu"
            files.append(output.write_collection(transient.output_times, extension))
        self._report_end(level, result, time.perf_counter() - start, files)
        return result

    @_changes_model
    def solve_eigenvalue(
        self,
        num_modes: int = 1,
        near: float = 0.0,
        tolerance: float = 1.0e-10,
        report: str = "full",
        output=None,
    ):
        """Compute eigenvalues and modes of the problem (see :mod:`dualmesh.modes`).

        Every field varies in time as :math:`\\hat{u}\\, e^{-\\lambda t}`, so the time derivatives define the eigenvalue :math:`\\lambda`, and the expressions may also use the symbol ``eigenvalue``, at most quadratically.
        The study returns the ``num_modes`` eigenvalues closest to ``near``, with their modes, in an :class:`~dualmesh.modes.EigenmodesResult`, and leaves the first mode in the solution.
        ``tolerance`` is the relative tolerance of the Arnoldi iteration.

        ``report`` sets what the study prints: ``"full"`` (the default: the problem with every parameter, the defaults marked, and the result), ``"summary"`` (one line that names the study, then the result) or ``"none"``.
        ``output`` is an :class:`~dualmesh.Output` group: the study writes the first mode once, at the end.
        Default None: no file."""
        from .modes import solve_modes

        level = report_level(report, "Problem.solve_eigenvalue")
        output = self._check_output(output, steady=True)
        self._build_physics()
        if level == "full":
            from .console import header

            print(header("eigenvalue study"))
            print(self.summary())
        elif level == "summary":
            print(
                f"dualmesh eigenvalue study: {self.num_global_dofs} unknowns, method {self.method}, shift-invert Arnoldi near {near:g}"
            )
        result = solve_modes(self, num_modes=num_modes, near=near, tolerance=tolerance)
        files = self._write_fields(output) if output is not None else []
        if level != "none":
            print(result.summary())
        if level == "full":
            for path in files:
                print(f"  wrote {path}")
        return result

    def solve_buckling(self, num_modes: int = 1, report: str = "full"):
        """Linear buckling about the solved state: (K + lambda K_sigma) x = 0, with K the
        stiffness of the undeformed body and K_sigma the geometric stiffness of the stresses of
        the solve before it, so that ``load_factors`` are the multiples of that load at which
        the body buckles.  The geometric stiffness is that of the geometrically nonlinear
        equations, so the solid takes ``large_deformation`` (a continuum) or ``von_karman``
        (a plate or a beam).  Returns a :class:`~dualmesh.modes.BucklingResult`; the solution
        is left as it was."""
        from .modes import solve_buckling

        level = report_level(report, "Problem.solve_buckling")
        self._build_physics()
        result = solve_buckling(self, num_modes=num_modes)
        if level != "none":
            print(result.summary())
        return result

    def set_time_step_callback(self, callback) -> None:
        """Call ``callback(time, problem)`` after every converged time step
        (after the post-processors have been evaluated)."""
        self._user_step_callback = callback
        self._install_step_callback()

    def _install_step_callback(self) -> None:
        user = self._user_step_callback
        extra = self._extra_step_callback
        if not self._postprocessors.postprocessors:
            if user is not None:
                self._problem.set_time_step_callback(user)
            return

        def step(time, core_problem):
            self._postprocessors.evaluate(self, time)
            if extra is not None:
                extra(time, core_problem)
            if user is not None:
                user(time, core_problem)

        self._problem.set_time_step_callback(step)

    # ---- post-processors -------------------------------------------------------
    def add_postprocessor(self, postprocessor_type: str, name: str, **parameters):
        """Add a post-processor, a scalar computed from the solution after every
        steady solve and every accepted time step: ``integral``, ``average``,
        ``point_value``, ``maximum``, ``minimum``, the flows through a
        boundary ``heat_flow``, ``reaction_force``, ``force``, ``flow_rate``
        and ``boundary_flux``, or ``kernel_boundary_flux`` (see
        :mod:`dualmesh.postprocessors`)."""
        from .postprocessors import create

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
        evaluation, with a header row of names.  In a distributed problem the first process writes it."""
        if self.rank == 0:
            self._postprocessors.write_csv(filename)

    def domain_volume(self) -> float:
        """The volume (area, length) of the domain, with the coordinate factor
        of the problem (2 pi r for axisymmetric, 4 pi r^2 for spherical)."""
        if self._domain_volume is None:
            probe = Problem(
                self._mesh, method="fem", coordinates=self.coordinates, distributed=False
            )
            probe.add_variable("one")
            probe.set_values("one", numpy.ones(len(probe.node_coordinates())))
            self._domain_volume = self._sum(float(probe.integrate("one")))
        return self._domain_volume

    def block_volume(self, blocks) -> float:
        """The volume (area, length) of the elements of ``blocks``, given by name or by id, with the
        coordinate factor of the problem, as :meth:`domain_volume` gives that of the domain."""
        names = self._mesh.block_names()
        wanted = set()
        for block in blocks:
            text = str(block)
            if text.lstrip("-").isdigit():
                wanted.add(int(text))
                continue
            ids = [number for number, name in names.items() if name == text]
            if not ids:
                raise _core.InputError(
                    f"There is no block '{text}'. The blocks are {sorted(names.values())}."
                )
            wanted.update(ids)
        probe = Problem(self._mesh, method="fem", coordinates=self.coordinates, distributed=False)
        volumes = probe.element_volumes()
        in_blocks = [block in wanted for _, _, block in self._mesh.cells()]
        return self._sum(float(numpy.sum(volumes[numpy.asarray(in_blocks, dtype=bool)])))

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

    def values(self, variable: str) -> numpy.ndarray:
        """Values of a variable, one per degree of freedom entity.

        These are nodal values indexed by node.  Use :meth:`entity_points` for the matching coordinates.
        """
        return self._problem.values(variable)

    def set_values(self, variable: str, values) -> None:
        self._problem.set_values(variable, list(numpy.asarray(values, dtype=float).ravel()))

    def set_element_field(self, name: str, values) -> None:
        """Set a named field with one value per element, which the property objects read
        by name (for instance a strain accumulated between time steps, read by
        a ``parsed_eigenstrain``).  Setting it again replaces
        the values, and the property objects see the new values at the next solve."""
        values = numpy.asarray(values, dtype=float).ravel()
        if len(values) != self._mesh.num_elements:
            raise ValueError(
                f"Element field '{name}': {len(values)} values for {self._mesh.num_elements} elements."
            )
        self._problem.set_element_field(name, list(values))

    def element_field(self, name: str) -> numpy.ndarray:
        """The values of a named element field."""
        return numpy.asarray(self._problem.element_field(name), dtype=float)

    def solution(self) -> numpy.ndarray:
        """The full solution vector (node-major, variable-minor)."""
        return self._problem.solution()

    def apply_initial_conditions(self) -> None:
        self._problem.apply_initial_conditions()

    def boundary_flux(self, variable: str, boundary: str, per_node: bool = False):
        """The flow of the flux of the equation of ``variable`` out of the body
        through ``boundary``: the heat leaving, the force of the body on its
        support, the volume flow leaving.  The physics name it:
        :meth:`~dualmesh.physics.HeatTransfer.heat_flow`,
        :meth:`~dualmesh.physics.SolidMechanics.reaction_force`,
        :meth:`~dualmesh.physics.FluidDynamics.force` and
        :meth:`~dualmesh.physics.FluidDynamics.flow_rate`.

        Where the boundary prescribes the value, the flow is the reaction: the
        residual of the equation at its nodes, which is the flux the condition
        supplies, conserved exactly.  Elsewhere it is the integral of the
        flux of the boundary conditions.  With ``per_node=True`` it returns
        ``{node: flow}``, the share of each node of the boundary.  A node on
        two boundaries of prescribed values (a corner) carries the reaction
        of both, so it appears in both.  Collective in a distributed problem:
        every process calls it (see :doc:`/theory/parallel`)."""
        if per_node:
            if self._distributed is not None:
                raise InputError(
                    "boundary_flux(per_node=True) is not available in a distributed problem."
                )
            return dict(self._problem.boundary_outflow(variable, boundary))
        if self._distributed is not None:
            return self._distributed.total_outflow(variable, boundary)
        return self._problem.total_outflow(variable, boundary)

    def maximum(self, variable: str) -> float:
        """The largest nodal value of a variable, over every process.  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        values = numpy.asarray(self.values(variable))
        local = float(values.max()) if values.size else -numpy.inf
        return _core.mpi_max(local) if self._distributed is not None else local

    def minimum(self, variable: str) -> float:
        """The smallest nodal value of a variable, over every process.  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        values = numpy.asarray(self.values(variable))
        local = float(values.min()) if values.size else numpy.inf
        return _core.mpi_min(local) if self._distributed is not None else local

    def _sum(self, value: float) -> float:
        """The sum of ``value`` over the processes of a distributed problem; ``value`` itself otherwise."""
        if self._distributed is None:
            return value
        return float(_core.mpi_sum([value])[0])

    def sample(self, variable: str, points) -> numpy.ndarray:
        """Interpolate a variable at arbitrary points (NaN outside the mesh).  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        points = numpy.atleast_2d(numpy.asarray(points, dtype=float))
        if points.shape[1] < 3:
            points = numpy.hstack([points, numpy.zeros((points.shape[0], 3 - points.shape[1]))])
        values = numpy.asarray(self._problem.sample(variable, [list(p) for p in points]))
        if self._distributed is None:
            return values
        # Every process interpolates in its own elements; a point on a partition boundary lies in elements of several processes, which give the same value, and the processes average what they found.
        found = ~numpy.isnan(values)
        totals = numpy.asarray(
            _core.mpi_sum(list(numpy.where(found, values, 0.0)) + list(found.astype(float)))
        )
        sums, counts = totals[: len(values)], totals[len(values) :]
        with numpy.errstate(invalid="ignore", divide="ignore"):
            return numpy.where(counts > 0, sums / counts, numpy.nan)

    def sample_gradient(self, variable: str, points) -> numpy.ndarray:
        """The gradient of a variable at arbitrary points, one row of three components each, in the
        element that holds the point (NaN outside the mesh).  Collective in a distributed problem."""
        points = numpy.atleast_2d(numpy.asarray(points, dtype=float))
        if points.shape[1] < 3:
            points = numpy.hstack([points, numpy.zeros((points.shape[0], 3 - points.shape[1]))])
        gradients = numpy.asarray(
            self._problem.sample_gradient(variable, [list(p) for p in points])
        )
        if self._distributed is None:
            return gradients
        # A point on a partition boundary lies in elements of several processes: the first that holds
        # it, in rank order, gives the gradient, so that every process returns the same one.
        found = ~numpy.isnan(gradients[:, 0])
        owner = numpy.asarray(
            [_core.mpi_min(float(self.rank) if held else numpy.inf) for held in found]
        )
        mine = found & (owner == self.rank)
        totals = numpy.asarray(
            _core.mpi_sum(list(numpy.where(mine[:, None], gradients, 0.0).ravel()))
        )
        totals = totals.reshape(gradients.shape)
        return numpy.where(numpy.isfinite(owner)[:, None], totals, numpy.nan)

    def nodes_where(self, predicate) -> list[int]:
        """Node indices whose coordinates satisfy ``predicate(x, y, z)``."""
        points = self._mesh.points()
        return [i for i, p in enumerate(points) if predicate(p[0], p[1], p[2])]

    def node_at(self, point, tolerance: float = 1e-9) -> int:
        """Index of the node closest to ``point`` (an error if none is within ``tolerance``)."""
        point = numpy.asarray(list(point) + [0.0] * (3 - len(point)), dtype=float)
        points = numpy.asarray(self._mesh.points())
        distances = numpy.linalg.norm(points - point, axis=1)
        index = int(numpy.argmin(distances))
        if distances[index] > tolerance:
            raise ValueError(
                f"No node within {tolerance} of {point.tolist()}; nearest is at {points[index].tolist()} (distance {distances[index]:.3e})."
            )
        return index

    def gradient_at_centroids(self, variable: str) -> numpy.ndarray:
        return numpy.asarray(self._problem.gradient_at_centroids(variable))

    def property_at_centroids(self, property_name: str) -> numpy.ndarray:
        return numpy.asarray(self._problem.property_at_centroids(property_name))

    def kernel_flux_at_centroids(self, kernel) -> numpy.ndarray:
        return numpy.asarray(self._problem.kernel_flux_at_centroids(_object_name(kernel)))

    def error_indicator(self, variable: str) -> numpy.ndarray:
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
        return numpy.asarray(self._problem.error_indicator(variable))

    def error_norms(
        self, variable: str, exact, exact_gradient=None, quadrature_points: int = 0
    ) -> tuple[float, float]:
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

        :math:`u_h` is the element interpolation of the nodal values.  The integrals use a Gauss rule of
        ``quadrature_points`` per direction on every element (zero chooses the
        polynomial order plus two, which over-integrates the leading term of
        the error) and include the coordinate factor.  The sum runs on several
        threads unless one of the functions is a Python callable.  A distributed
        problem sums the squares over every process, so the norms are those of
        the whole domain; the call is collective (every process makes it).
        """
        l2, h1 = self._problem.error_norms(
            variable,
            _as_function(exact),
            None if exact_gradient is None else [_as_function(g) for g in exact_gradient],
            quadrature_points,
        )
        if self._distributed is None:
            return float(l2), float(h1)
        # Each process integrates its own elements: the squares add.
        squares = _core.mpi_sum([float(l2) ** 2, float(h1) ** 2])
        return float(numpy.sqrt(squares[0])), float(numpy.sqrt(squares[1]))

    def linear_system(self):
        r"""The residual and the Jacobian of the steady problem at the current
        solution, as ``(residual, jacobian)``.

        This is the system one Newton step of :meth:`solve` solves,
        :math:`J\,\delta U = -R`: the prescribed boundary values are written
        into the solution first, and the rows (and columns) of the prescribed
        degrees of freedom are dropped but for the diagonal, which takes the
        mean size of the diagonal of their equation.  ``residual``
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

    def boundary_area(self, boundary) -> float:
        r"""The area of one side set, or of a list of side sets: the length of
        the boundary in two dimensions and its area in three.  In axisymmetric and
        spherical coordinates the area includes the coordinate factor, i.e., it
        is the area :math:`\int 2 \pi r \, \mathrm{d}s` of the surface of
        revolution.  It is the area :math:`A` over which the ``total_force`` of a
        ``traction`` is spread."""
        names = [boundary] if isinstance(boundary, str) else list(boundary)
        return self._problem.boundary_area(names)

    def integrate(self, variable: str) -> float:
        """The integral of a variable over the domain, with the coordinate
        factor of the problem.  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        return self._sum(float(self._problem.integrate(variable)))

    def element_integrals(self, variable: str) -> numpy.ndarray:
        """The integral of a variable over every element (one value per
        element, with the coordinate factor of the problem), e.g., for the
        average of a flux over a region."""
        return numpy.asarray(self._problem.element_integrals(variable), dtype=float)

    def element_volumes(self) -> numpy.ndarray:
        """The volume of every element, with the coordinate factor of the
        problem (e.g., :math:`2 \\pi r` in axisymmetric coordinates)."""
        return numpy.asarray(self._problem.element_integrals(""), dtype=float)

    def kernel_boundary_flux(self, kernel, boundary: str) -> float:
        """The integral of the normal flux of one kernel over a side set.  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        return self._sum(float(self._problem.kernel_boundary_flux(_object_name(kernel), boundary)))

    # ---- output ----------------------------------------------------------
    def write_vtu(
        self, filename: str, cell_properties: Sequence[str] = (), fields: Sequence[str] = ()
    ) -> None:
        """Write a VTK unstructured grid (readable by ParaView and VisIt).  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`).

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
        return (
            self._distributed.num_owned_dofs()
            if self._distributed is not None
            else self.num_active_dofs()
        )

    @property
    def num_global_dofs(self) -> int:
        """Degrees of freedom of the whole problem."""
        return (
            self._distributed.num_global_dofs()
            if self._distributed is not None
            else self.num_active_dofs()
        )

    def gathered_values(self, variable: str) -> numpy.ndarray:
        """Values of a variable at every node of the whole mesh, on every
        process.  A distributed problem allocates one vector of the global
        size per process, so this suits tests and small problems, and a large
        run writes its result with :meth:`write_vtu`.  For a serial problem it
        equals :meth:`values`.  Collective in a distributed problem: every process calls it (see :doc:`/theory/parallel`)."""
        if self._distributed is not None:
            return numpy.asarray(self._distributed.gathered_values(variable))
        return self.values(variable)

    def gathered_error_indicator(self, variable: str) -> numpy.ndarray:
        """:meth:`error_indicator` of every element of the whole mesh, in its numbering, on every
        process.  Collective in a distributed problem; in a serial one it equals
        :meth:`error_indicator`."""
        local = self._problem.error_indicator(variable)
        if self._distributed is not None:
            return numpy.asarray(self._distributed.gathered_element_values(local))
        return numpy.asarray(local)

    def set_gathered_values(self, variable: str, values) -> None:
        """Set a variable from its values at every node of the whole mesh, in its numbering, on
        every process; in a serial problem it equals :meth:`set_values`."""
        values = [float(v) for v in numpy.asarray(values, dtype=float).ravel()]
        if self._distributed is not None:
            self._distributed.set_gathered_values(variable, values)
        else:
            self._problem.set_values(variable, values)

    def partition_summary(self) -> str:
        """One line describing the partition of a distributed problem."""
        return (
            self._distributed.summary()
            if self._distributed is not None
            else "serial problem (one process)"
        )

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
        row for each node, with its coordinates.  The header names
        every column with its unit, for example ``temperature (K)``.
        ``variables`` selects the variables (default: all of them)."""
        from .tables import column_header

        names = list(variables) or [
            self._problem.variable_name(i) for i in range(self._problem.num_variables)
        ]
        points = numpy.asarray(self.node_coordinates())
        dimension = self._mesh.space_dimension
        columns = [points[:, i] for i in range(dimension)]
        columns += [self.values(name) for name in names]
        header = ",".join(
            [column_header(axis, "m") for axis in ("x", "y", "z")[:dimension]]
            + [column_header(name, self.variable_unit(name)) for name in names]
        )
        numpy.savetxt(
            filename, numpy.column_stack(columns), delimiter=",", header=header, comments=""
        )

    def summary(self, parameters: bool = True) -> str:
        """A report of the problem: method, mesh, variables and objects, with
        every parameter of every object and its value, marked ``(default)``
        when it was not set.  See :mod:`dualmesh.console`."""
        from .console import problem_report

        self._build_physics()
        return problem_report(self, parameters=parameters)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<dualmesh.Problem method={self.method} {self._mesh.num_nodes} nodes>"


@dataclass
class SensitivityResult(ResultTables):
    """The outcome of a sensitivity study: d(objective)/d(parameter) of every
    named parameter, by one adjoint solve of :meth:`Problem.sensitivities`.

    ``values`` maps a parameter name to its derivative, ``tables
    ["sensitivities"]`` is the table of them, and ``summary()``,
    ``to_dict()``, ``write_json(path)`` and ``write_csv(path)`` follow the
    result protocol."""

    objective: tuple  # (boundary, variable); an empty boundary: the domain
    values: dict

    def description(self) -> str:
        """The objective in words, for the titles."""
        boundary, variable = self.objective
        if not boundary:
            return f"integral of {variable} over the domain"
        return f"integral of {variable} over the boundary '{boundary}'"

    @property
    def tables(self) -> dict:
        from .tables import Table

        rows = [[name, value] for name, value in self.values.items()]
        return {
            "sensitivities": Table(
                ["parameter", "d objective / d parameter"],
                ["", ""],
                rows,
                title=f"Sensitivities of the {self.description()}",
            )
        }

    def summary(self) -> str:
        return self.tables["sensitivities"].format()

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        return {
            "study": "sensitivity",
            "objective": self.description(),
            "sensitivities": dict(self.values),
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)
