# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The eigenvalue study of any problem: eigenvalues and modes.

Every field is taken to vary in time as :math:`u(t) = \hat{u}\, e^{-\lambda t}`,
so a time derivative :math:`d\, \partial u / \partial t` becomes
:math:`-\lambda d\, \hat{u}`.  The expressions may also use the symbol
``eigenvalue`` for :math:`\lambda` directly, for example a source
``eigenvalue*nu_sigma_f*phi`` for the criticality of a reactor.  The study
linearises the equations at the current solution: their Jacobian at
:math:`\lambda` is

.. math::

   \mathbf{J}(\lambda) = \mathbf{K} + \lambda (\mathbf{B}_1 - \mathbf{M}) + \lambda^2 \mathbf{B}_2 ,

where :math:`\mathbf{K}` is the Jacobian of the steady terms with
:math:`\lambda = 0`, :math:`\mathbf{M}` the Jacobian of the time derivatives,
and :math:`\mathbf{B}_1` and :math:`\mathbf{B}_2` the linear and quadratic
parts of the terms that use ``eigenvalue``.  Three assemblies, at
:math:`\lambda = 0`, 1 and -1, give :math:`\mathbf{B}_1` and
:math:`\mathbf{B}_2` exactly, and a fourth at :math:`\lambda = 2` checks
that the terms are at most quadratic in :math:`\lambda`.  The eigenvalues are
those :math:`\lambda` for which :math:`\mathbf{J}(\lambda) \hat{\mathbf{u}} = 0`
has a nonzero solution, with the prescribed values of the boundary conditions
set to zero and source terms that do not depend on the fields left out.  The study computes the eigenvalues closest to a
given value by shift-invert Arnoldi iteration (ARPACK through SciPy); a
quadratic problem is solved in its companion form of twice the size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .tables import ResultTables

__all__ = ["EigenmodesResult", "solve_modes"]


@dataclass
class EigenmodesResult(ResultTables):
    """The outcome of an eigenvalue study: the eigenvalues, closest to
    ``near`` first, and the modes.

    ``modes[k]`` maps each variable to its values in mode ``k``, scaled so
    that the largest value of the mode is 1.  ``summary()`` is the report,
    ``tables["eigenvalues"]`` the table of the eigenvalues, and
    ``write_csv(path)`` and ``write_json(path)`` write them."""

    eigenvalues: numpy.ndarray
    near: float
    num_dofs: int
    residual_norms: numpy.ndarray
    quadratic: bool
    #: A vibration: the only time term is a second time derivative, so the eigenvalues are
    #: lambda = i omega, with omega the angular frequency of a mode.
    vibration: bool = False
    modes: list = field(default_factory=list, repr=False)

    @property
    def frequencies(self) -> numpy.ndarray:
        """The natural frequencies, Hz, of a problem with inertia: |Im lambda| / (2 pi), for
        the time dependence e^(-lambda t)."""
        return numpy.abs(numpy.imag(numpy.asarray(self.eigenvalues, dtype=complex))) / (
            2.0 * numpy.pi
        )

    @property
    def tables(self) -> dict:
        from .tables import Table

        rows = [
            [k + 1, float(numpy.real(value)), float(numpy.imag(value)), float(norm)]
            for k, (value, norm) in enumerate(zip(self.eigenvalues, self.residual_norms))
        ]
        columns = ["mode", "eigenvalue_real", "eigenvalue_imaginary", "relative_residual"]
        return {"eigenvalues": Table(columns, ["", "", "", ""], rows, title="Eigenvalues")}

    def summary(self) -> str:
        from .tables import Table

        head = Table(
            ["quantity", "value"],
            None,
            [
                ["modes", len(self.eigenvalues)],
                ["closest to", f"{self.near:g}"],
                ["dependence on the eigenvalue", "quadratic" if self.quadratic else "linear"],
                ["unknowns", self.num_dofs],
            ],
            title="Eigenvalue study",
        )
        return "\n\n".join([head.format(), self.tables["eigenvalues"].format()])

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        return {
            "study": "eigenvalue",
            "near": self.near,
            "quadratic": self.quadratic,
            "num_dofs": self.num_dofs,
            "eigenvalues_real": [float(numpy.real(v)) for v in self.eigenvalues],
            "eigenvalues_imaginary": [float(numpy.imag(v)) for v in self.eigenvalues],
            "relative_residuals": [float(r) for r in self.residual_norms],
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)


def _operator(problem, steady_terms: bool, time_kernels: bool, eigenvalue: float, order: int = 1):
    import scipy.sparse as sparse

    values, indices, indptr, mask = problem._problem._operator(
        steady_terms, time_kernels, eigenvalue, order
    )
    n = len(indptr) - 1
    return sparse.csc_matrix((values, indices, indptr), shape=(n, n)), numpy.asarray(
        mask, dtype=bool
    )


def _norm(matrix) -> float:
    return float(abs(matrix).max()) if matrix.nnz else 0.0


def _mode_values(problem, index, size, x, names) -> dict:
    """The values of every variable in a mode ``x`` of the free unknowns."""
    parts = {}
    for part, values in (("real", numpy.real(x)), ("imaginary", numpy.imag(x))):
        full = numpy.zeros(size)
        full[index] = values
        problem._problem.set_solution(list(full))
        parts[part] = {name: numpy.asarray(problem.values(name)) for name in names}
    if not numpy.iscomplexobj(x):
        return parts["real"]
    return {name: parts["real"][name] + 1j * parts["imaginary"][name] for name in names}


def solve_modes(
    problem, num_modes: int = 1, near: float = 0.0, tolerance: float = 1.0e-10
) -> EigenmodesResult:
    """See :meth:`dualmesh.Problem.solve_eigenvalue`."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    if problem.is_distributed:
        raise ValueError("solve_modes: the eigenvalue study runs on one process at present.")
    if num_modes < 1:
        raise ValueError("solve_modes: num_modes must be at least 1.")
    problem.initialize()
    stiffness, mask = _operator(problem, True, False, 0.0)
    mass, _ = _operator(problem, False, True, 0.0)
    # The second time derivatives: d2u/dt2 = lambda^2 u.
    inertia, _ = _operator(problem, False, True, 0.0, order=2)
    plus, _ = _operator(problem, True, False, 1.0)
    minus, _ = _operator(problem, True, False, -1.0)
    two, _ = _operator(problem, True, False, 2.0)
    linear = (plus - minus) / 2.0
    quadratic = (plus + minus) / 2.0 - stiffness
    # The parts in the symbol eigenvalue are differences of assemblies that hold the steady
    # terms, so their round-off is of the size of those; the time derivatives are assembled
    # alone, and exactly zero where there are none.  Each is measured in its own units.
    steady = max(_norm(stiffness), _norm(plus), _norm(minus), _norm(two), 1e-300)
    if _norm(two - (stiffness + 2.0 * linear + 4.0 * quadratic)) > 1e-9 * steady:
        raise ValueError(
            "solve_modes: the equations depend on the eigenvalue more than quadratically. Write the terms that use 'eigenvalue' as at most quadratic in it."
        )
    if _norm(linear) <= 1e-12 * steady:
        linear = 0.0 * linear
    if _norm(quadratic) <= 1e-12 * steady:
        quadratic = 0.0 * quadratic
    free = ~mask
    index = numpy.flatnonzero(free)
    restrict = sparse.csc_matrix(
        (numpy.ones(len(index)), (numpy.arange(len(index)), index)), shape=(len(index), len(mask))
    )
    K = (restrict @ stiffness @ restrict.T).tocsc()
    D = (restrict @ (linear - mass) @ restrict.T).tocsc()
    E = (restrict @ (quadratic + inertia) @ restrict.T).tocsc()
    n = K.shape[0]
    is_quadratic = _norm(E) > 0.0
    if not is_quadratic and _norm(D) == 0.0:
        raise ValueError(
            "solve_modes: the problem has no time derivative and no term that uses 'eigenvalue', so it defines no eigenvalue."
        )
    undamped = is_quadratic and _norm(D) == 0.0
    # The companion form is solved in the time scale s = sqrt(|K| / |E|) of the problem, lambda
    # = s lambda~, with K, s D and s^2 E divided by |K|: its blocks are then of one size in any
    # units (in SI a steel structure has |K| / |E| near 1e11, and the unscaled form lost
    # six digits of the damping).
    time_scale = 1.0
    if undamped:
        # K x + lambda^2 E x = 0: a problem in mu = lambda^2 of the size of K, K x = mu (-E) x.
        A, B = K, (-E).tocsc()
    elif is_quadratic:
        # Companion form: with y = lambda x, [0 I; -K -D] [x; y] = lambda [I 0; 0 E] [x; y].
        size_k = max(_norm(K), 1e-300)
        time_scale = numpy.sqrt(size_k / _norm(E))
        identity = sparse.identity(n, format="csc")
        A = sparse.bmat([[None, identity], [-K / size_k, -D * (time_scale / size_k)]], format="csc")
        B = sparse.bmat([[identity, None], [None, E * (time_scale**2 / size_k)]], format="csc")
    else:
        # K x + lambda D x = 0, that is K x = lambda C x with C = -D.
        A, B = K, (-D).tocsc()
    size = A.shape[0]
    if num_modes > size - 2:
        raise ValueError(
            f"solve_modes: num_modes is {num_modes}, and the problem has {n} free unknowns. Ask for fewer modes."
        )
    # The Arnoldi iteration computes more eigenvalues than asked, so that a repeated eigenvalue
    # is found with its multiplicity; the closest num_modes are returned.
    count = min(max(2 * num_modes, num_modes + 5), size - 2)
    shift = near**2 if undamped else near / time_scale
    factor = linalg.splu((A - shift * B).tocsc())
    shifted = linalg.LinearOperator((size, size), matvec=lambda v: factor.solve(B @ v), dtype=float)
    mu, vectors = linalg.eigs(shifted, k=count, which="LM", tol=tolerance)
    eigenvalues = (shift + 1.0 / mu) * time_scale
    if undamped:
        # lambda = sqrt(mu), with the root of non-negative imaginary part (a vibration e^(i w t)).
        eigenvalues = numpy.sqrt(eigenvalues.astype(complex))
        eigenvalues = numpy.where(numpy.imag(eigenvalues) < 0, -eigenvalues, eigenvalues)
    order = numpy.argsort(numpy.abs(eigenvalues - near))[:num_modes]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if not numpy.iscomplexobj(eigenvalues) or numpy.all(
        numpy.abs(numpy.imag(eigenvalues)) <= 1e-10 * numpy.abs(eigenvalues)
    ):
        eigenvalues = numpy.real(eigenvalues)
    names = list(problem._problem.variable_names())
    modes, residuals = [], []
    for k in range(len(eigenvalues)):
        x = vectors[:n, k]
        if numpy.all(numpy.abs(numpy.imag(x)) <= 1e-10 * numpy.abs(x).max()):
            x = numpy.real(x)
        x = x / x[numpy.argmax(numpy.abs(x))]
        value = eigenvalues[k]
        residual = (K + value * D + value**2 * E) @ x
        reference = (
            numpy.abs(K @ x).max()
            + abs(value) * numpy.abs(D @ x).max()
            + abs(value) ** 2 * numpy.abs(E @ x).max()
        )
        residuals.append(float(numpy.abs(residual).max() / max(reference, 1e-300)))
        modes.append(_mode_values(problem, index, len(mask), x, names))
    # The solution holds the real part of the first mode.
    first = numpy.zeros(len(mask))
    first[index] = numpy.real(vectors[:n, 0] / vectors[numpy.argmax(numpy.abs(vectors[:n, 0])), 0])
    problem._problem.set_solution(list(first))
    return EigenmodesResult(
        eigenvalues=numpy.asarray(eigenvalues),
        near=float(near),
        num_dofs=n,
        residual_norms=numpy.asarray(residuals),
        quadratic=bool(is_quadratic),
        vibration=bool(undamped),
        modes=modes,
    )


@dataclass
class BucklingResult(ResultTables):
    """The outcome of a buckling study: the critical load factors, smallest first, and the
    buckling modes.

    ``load_factors[k]`` is the multiple of the load of the solve before the study at which
    mode ``k`` buckles; ``modes[k]`` maps each variable to its values in that mode, scaled so
    that its largest value is 1."""

    load_factors: numpy.ndarray
    residual_norms: numpy.ndarray
    modes: list = field(default_factory=list, repr=False)

    @property
    def tables(self) -> dict:
        from .tables import Table

        rows = [
            [k + 1, float(value), float(norm)]
            for k, (value, norm) in enumerate(zip(self.load_factors, self.residual_norms))
        ]
        return {
            "load_factors": Table(
                ["mode", "load_factor", "relative_residual"], ["", "", ""], rows, title="Buckling"
            )
        }

    def summary(self) -> str:
        return self.tables["load_factors"].format()

    def __str__(self) -> str:
        return self.summary()


def solve_buckling(problem, num_modes: int = 1, tolerance: float = 1.0e-10) -> BucklingResult:
    """See :meth:`dualmesh.Problem.solve_buckling`."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    if problem.is_distributed:
        raise ValueError("solve_buckling: the buckling study runs on one process at present.")
    problem.initialize()
    # The stiffness at the solved state is K + K_sigma, with K_sigma the geometric stiffness of
    # its stresses; at the undeformed state it is K.  Both are assembled without the rows of the
    # prescribed values, which are removed here.
    solved = numpy.asarray(problem._problem.solution())
    loaded, mask = _operator(problem, True, False, 0.0)
    problem._problem.set_solution([0.0] * len(solved))
    unloaded, _ = _operator(problem, True, False, 0.0)
    problem._problem.set_solution(list(solved))
    geometric = loaded - unloaded
    if _norm(geometric) <= 1e-9 * _norm(unloaded):
        raise ValueError(
            "solve_buckling: the problem has no geometric stiffness at its solved state. Solve the load first, with large_deformation (a continuum) or von_karman (a plate or a beam)."
        )
    index = numpy.flatnonzero(~mask)
    restrict = sparse.csc_matrix(
        (numpy.ones(len(index)), (numpy.arange(len(index)), index)), shape=(len(index), len(mask))
    )
    K = (restrict @ unloaded @ restrict.T).tocsc()
    G = (restrict @ geometric @ restrict.T).tocsc()
    n = K.shape[0]
    if num_modes > n - 2:
        raise ValueError(
            f"solve_buckling: num_modes is {num_modes}, and the problem has {n} free unknowns."
        )
    # (K + lambda K_sigma) x = 0: with K factored, -K^-1 K_sigma x = x / lambda, whose largest
    # values are the smallest load factors.
    factor = linalg.splu(K)
    operator = linalg.LinearOperator((n, n), matvec=lambda v: -factor.solve(G @ v), dtype=float)
    count = min(max(2 * num_modes, num_modes + 5), n - 2)
    inverse, vectors = linalg.eigs(operator, k=count, which="LR", tol=tolerance)
    keep = numpy.real(inverse) > 0
    factors = 1.0 / numpy.real(inverse[keep])
    vectors = numpy.real(vectors[:, keep])
    order = numpy.argsort(factors)[:num_modes]
    factors, vectors = factors[order], vectors[:, order]
    names = list(problem._problem.variable_names())
    modes, residuals = [], []
    for k in range(len(factors)):
        x = vectors[:, k] / vectors[numpy.argmax(numpy.abs(vectors[:, k])), k]
        residual = (K + factors[k] * G) @ x
        reference = numpy.abs(K @ x).max() + abs(factors[k]) * numpy.abs(G @ x).max()
        residuals.append(float(numpy.abs(residual).max() / max(reference, 1e-300)))
        modes.append(_mode_values(problem, index, len(mask), x, names))
    problem._problem.set_solution(list(solved))
    return BucklingResult(
        load_factors=numpy.asarray(factors), residual_norms=numpy.asarray(residuals), modes=modes
    )
