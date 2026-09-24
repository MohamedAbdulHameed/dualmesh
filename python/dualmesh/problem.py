# SPDX-License-Identifier: LGPL-2.1-or-later
"""The :class:`Problem` class: variables, objects, solvers, and results."""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence

import numpy as np

from . import _core

SolveResult = _core.SolveResult


def _as_function(value):
    """A number, an expression string, a compiled function or a callable, as
    something the extension accepts where it expects a function."""
    if isinstance(value, str):
        return _core.ParsedFunction(value)
    return value


def _solver_options(**kwargs) -> _core.SolverOptions:
    options = _core.SolverOptions()
    known = {
        "nonlinear_solver",
        "max_iterations",
        "relative_tolerance",
        "absolute_tolerance",
        "step_tolerance",
        "relaxation",
        "load_factors",
        "linear_solver",
        "preconditioner",
        "gmres_restart",
        "linear_tolerance",
        "linear_max_iterations",
        "verbose",
        "error_on_divergence",
    }
    for key, value in kwargs.items():
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise TypeError(
                f"Unknown solver option '{key}'.{hint} Known options: {', '.join(sorted(known))}."
            )
        setattr(options, key, value)
    return options


class Problem:
    """A boundary value problem discretized on a primal mesh and its dual mesh.

    Parameters
    ----------
    mesh:
        The primal mesh of finite elements.
    method:
        The discretization.  ``"dmcdm"`` is the dual mesh control domain method
        (the default); ``"fem"`` is the Galerkin finite element method;
        ``"hfvm"`` is the vertex-centred finite volume method, which uses the
        same control domains as the dual mesh method but two-point gradients at
        their interfaces (the half-control volume formulation of Reddy,
        Chapter 3); ``"zfvm"`` is the cell-centred finite volume method, with
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
    """

    def __init__(
        self,
        mesh,
        method: str = "dmcdm",
        coordinates: str = "cartesian",
        boundary_gradient: str = "first_order",
    ):
        self._problem = _core.Problem(mesh, method, coordinates)
        self._problem.set_boundary_gradient(boundary_gradient)
        self._mesh = mesh
        self._objects: list = []  # keeps Python-defined objects alive
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

    @classmethod
    def _wrap(cls, core_problem, mesh, method: str, coordinates: str) -> Problem:
        """Wrap an existing extension-level problem (used by the distributed
        solver, which creates the rank-local problem itself)."""
        self = cls.__new__(cls)
        self._problem = core_problem
        self._mesh = mesh
        self._objects = []
        self.method = method
        self.coordinates = coordinates
        self.boundary_gradient = "first_order"
        return self

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
    def add_variable(self, name: str, blocks: Sequence[str] = (), initial_condition=None) -> int:
        """Add a nodal unknown; returns its index."""
        return self._problem.add_variable(name, list(blocks), initial_condition)

    def add_function(self, name: str, function) -> None:
        """Register a named function of ``(x, y, z, t)`` (or a constant)."""
        self._problem.add_function(name, function)

    def _add(self, adder, object_or_type, name, kwargs):
        if isinstance(object_or_type, str):
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
        """Add a boundary condition (essential or natural)."""
        if isinstance(condition, str):
            category = _core.object_category(condition)
            if category == "NodalBC":
                return self._problem.add_object(condition, name or "", **parameters)
            if category == "BoundaryCondition":
                return self._problem.add_object(condition, name or "", **parameters)
            raise ValueError(f"'{condition}' is a {category}, not a boundary condition.")
        if isinstance(condition, _core.NodalBC):
            return self._add(self._problem.add_nodal_bc, condition, name, parameters)
        return self._add(self._problem.add_integrated_bc, condition, name, parameters)

    def add_material(self, material, name: str | None = None, **parameters):
        """Add a material (a provider of named properties)."""
        return self._add(self._problem.add_material, material, name, parameters)

    def add_point_source(self, source="PointSource", name: str | None = None, **parameters):
        """Add a concentrated nodal source (point force or point heat source)."""
        return self._add(self._problem.add_nodal_load, source, name, parameters)

    add_nodal_load = add_point_source

    def initialize(self) -> None:
        """Resolve all objects (called automatically by :meth:`solve`)."""
        self._problem.initialize()

    # ---- solving ---------------------------------------------------------
    def solve(self, **options) -> SolveResult:
        r"""Solve the steady problem.

        Keyword arguments are solver options: ``nonlinear_solver`` (``"newton"``,
        ``"picard"``, or ``"linear"``), ``max_iterations``, ``relative_tolerance``,
        ``absolute_tolerance``, ``step_tolerance``, ``relaxation`` (the
        acceleration parameter of direct iteration), ``load_factors`` (load
        stepping), ``linear_solver`` (``"automatic"``, the default, ``"lu"``,
        ``"bicgstab"``, ``"gmres"`` or ``"cg"``), ``preconditioner`` (``"ilu"``,
        the default, ``"ilut"``, ``"jacobi"`` or ``"none"``),
        ``linear_tolerance``, ``linear_max_iterations``, ``gmres_restart``,
        ``verbose``, and ``error_on_divergence``.

        ``"automatic"`` factorises the system directly where that is cheap,
        which is always in one dimension, up to :math:`10^5` unknowns in two
        and up to a few thousand in three, and otherwise uses BiCGSTAB
        preconditioned by an incomplete LU factorisation, falling back to the
        direct solver if the iteration does not converge.  On a
        three-dimensional mesh the iteration is typically ten to fifty times
        faster than the direct solver, because the direct factors of a 3D
        problem fill in far more.
        """
        return self._problem.solve_steady(_solver_options(**options))

    def solve_transient(
        self,
        end_time: float,
        dt: float,
        start_time: float = 0.0,
        theta: float = 1.0,
        output_interval: int = 0,
        output_file_base: str = "",
        time_stepper: str = "fixed",
        dt_min: float = 0.0,
        dt_max: float = 0.0,
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

        so ``theta=1`` is backward Euler, which is unconditionally stable and
        first-order accurate; ``theta=0.5`` is the Crank-Nicolson, or midpoint,
        rule, which is unconditionally stable and second-order accurate but can
        ring on a sharp transient; and ``theta=0`` is forward Euler, which is
        explicit in the steady terms and is stable only below a critical step.

        Parameters
        ----------
        end_time, dt, start_time:
            The interval to cover and the step to take.  With an adaptive
            stepper ``dt`` is the first step rather than every step.
        theta:
            The weight above.
        output_interval, output_file_base:
            Write a ``.vtu`` file every so many accepted steps.
        time_stepper:
            ``"fixed"`` keeps the step, shortening only the last one so that
            the run lands exactly on ``end_time``.

            ``"error"`` chooses the step from an estimate of the local
            truncation error.  Each interval is advanced twice, once with one
            step and once with two half steps, and the difference between the
            two answers estimates the error of the coarse one by Richardson
            extrapolation.  A step whose relative error exceeds
            ``error_tolerance`` is discarded and retried with a smaller step;
            an accepted step is followed by the largest step the estimate
            allows.  The answer that is kept is the accurate one, from the two
            half steps.  The estimate costs three nonlinear solves per accepted
            step, so use this when accuracy in time is what matters.

            ``"iteration"`` chooses the step from how hard the nonlinear solver
            worked.  A step that converged in fewer than
            ``optimal_iterations - iteration_window`` iterations is followed by
            a larger one, a step that needed more than
            ``optimal_iterations + iteration_window`` by a smaller one, and a
            step that failed to converge is discarded and retried.  It costs
            nothing beyond the solve and is the right choice when the
            difficulty is the nonlinearity rather than the accuracy.
        dt_min, dt_max:
            Bounds on the step.  A run that has to go below ``dt_min`` is
            reported as a failure rather than grinding to a halt.  Zero means
            ``dt`` divided by one million, and the whole interval,
            respectively.
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
            Passed to the nonlinear solver of every step; see :meth:`solve`.

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
        transient.dt = dt
        transient.theta = theta
        transient.output_interval = output_interval
        transient.output_file_base = output_file_base
        transient.time_stepper = time_stepper
        transient.dt_min = dt_min
        transient.dt_max = dt_max
        transient.growth_factor = growth_factor
        transient.cutback_factor = cutback_factor
        transient.error_tolerance = error_tolerance
        transient.optimal_iterations = optimal_iterations
        transient.iteration_window = iteration_window
        transient.max_rejected_steps = max_rejected_steps
        return self._problem.solve_transient(transient, _solver_options(**options))

    def set_time_step_callback(self, callback) -> None:
        """Call ``callback(time, problem)`` after every converged time step."""
        self._problem.set_time_step_callback(callback)

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

        For every method but ``"zfvm"`` these are nodal values indexed by node;
        for ``"zfvm"`` they are cell values followed by boundary face values.
        Use :meth:`entity_points` for the matching coordinates.
        """
        return self._problem.values(variable)

    def set_values(self, variable: str, values) -> None:
        self._problem.set_values(variable, list(np.asarray(values, dtype=float).ravel()))

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
            raise ValueError(
                f"No node within {tolerance} of {point.tolist()}; nearest is at "
                f"{points[index].tolist()} (distance {distances[index]:.3e})."
            )
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
        difference measures the error in the computed one; this is the
        estimator of Zienkiewicz and Zhu (1987).

        It is an indicator, not a bound.  It says which elements carry most of
        the error, which is what :func:`~dualmesh.mark_by_fraction`
        and its relatives need, and it does not certify the size of the error.
        """
        return np.asarray(self._problem.error_indicator(variable))

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
        the same, one per component; missing components are taken as zero, and
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
        l2, h1 = self._problem.error_norms(
            variable,
            _as_function(exact),
            None if exact_gradient is None else [_as_function(g) for g in exact_gradient],
            quadrature_points,
        )
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
        residual, values, indices, indptr = self._problem._linear_system()
        n = residual.shape[0]
        jacobian = sparse.csc_matrix((values, indices, indptr), shape=(n, n))
        return residual, jacobian

    def integrate(self, variable: str) -> float:
        return self._problem.integrate(variable)

    def boundary_flux_integral(self, kernel_name: str, boundary: str) -> float:
        return self._problem.boundary_flux_integral(kernel_name, boundary)

    # ---- output ----------------------------------------------------------
    def write_vtu(self, filename: str, cell_properties: Sequence[str] = ()) -> None:
        """Write a VTK unstructured grid (readable by ParaView and VisIt)."""
        self._problem.write_vtu(filename, list(cell_properties))

    def write_mesh_file(self, filename: str, file_format: str | None = None) -> None:
        """Write the mesh and all nodal fields through meshio (Exodus, VTU, ...)."""
        from .meshing import write_mesh

        fields = {}
        for index in range(self._problem.num_variables):
            name = self._problem.variable_name(index)
            fields[name] = self.values(name)
        write_mesh(self._mesh, filename, file_format=file_format, **fields)

    def write_csv(self, filename: str, variables: Sequence[str] = ()) -> None:
        """Write nodal coordinates and values as comma-separated values."""
        names = list(variables) or [
            self._problem.variable_name(i) for i in range(self._problem.num_variables)
        ]
        points = np.asarray(self._mesh.points())
        columns = [points[:, i] for i in range(self._mesh.dimension)]
        columns += [self.values(name) for name in names]
        header = ",".join(["x", "y", "z"][: self._mesh.dimension] + names)
        np.savetxt(filename, np.column_stack(columns), delimiter=",", header=header, comments="")

    def summary(self) -> str:
        return self._problem.summary()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<dualmesh.Problem method={self.method} {self._mesh.num_nodes} nodes>"
