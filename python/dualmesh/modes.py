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

   \mathbf{J}(\lambda) = \mathbf{K} + \lambda (\mathbf{B}_1 - \mathbf{M}) + \lambda^2 (\mathbf{B}_2 + \mathbf{M}_2) ,

where :math:`\mathbf{K}` is the Jacobian of the steady terms with
:math:`\lambda = 0`, :math:`\mathbf{M}` the Jacobian of the first time
derivatives, :math:`\mathbf{M}_2` that of the second time derivatives, which
become :math:`\lambda^2 \hat{u}`,
and :math:`\mathbf{B}_1` and :math:`\mathbf{B}_2` the linear and quadratic
parts of the terms that use ``eigenvalue``.  Three assemblies, at
:math:`\lambda = 0`, 1 and -1, give :math:`\mathbf{B}_1` and
:math:`\mathbf{B}_2` exactly, and a fourth at :math:`\lambda = 2` checks
that the terms are at most quadratic in :math:`\lambda`.  The eigenvalues are
those :math:`\lambda` for which :math:`\mathbf{J}(\lambda) \hat{\mathbf{u}} = 0`
has a nonzero solution, with the prescribed values of the boundary conditions
set to zero and source terms that do not depend on the fields left out.  The study computes the eigenvalues closest to a
given value by shift-invert Arnoldi iteration (ARPACK through SciPy). A
quadratic problem without a first-order term,
:math:`\mathbf{B}_1 - \mathbf{M} = 0`, is solved for :math:`\mu = \lambda^2`,
and each :math:`\mu` gives the pair :math:`\lambda = \pm\sqrt{\mu}`, of
which the study reports one. Any other quadratic problem is solved in its
companion form of twice the size.

The given value is a value of :math:`\lambda`, with one exception. In a
vibration, the second time derivative is the only time term and no term uses
``eigenvalue``, so :math:`\mathbf{J}(\lambda) = \mathbf{K} + \lambda^2 \mathbf{M}_2`.
Its eigenvalues are :math:`\lambda = i \omega`, with :math:`\omega` the angular
frequency in 1/s, and the given value is an :math:`\omega`. The study reports
the root :math:`i \omega` of non-negative imaginary part. In any other problem
without a first-order term, the study reports the root closer to the given
value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .tables import ResultTables, Table

__all__ = ["EigenmodesResult", "solve_modes"]


#: A floor that keeps a quotient finite where its divisor is zero.
_TINY = 1e-300
#: The relative size, against the steady terms, of the difference between the Jacobian at
#: lambda = 2 and its quadratic through lambda = 0, 1 and -1, above which the equations depend on
#: lambda more than quadratically.
_QUADRATIC_TOLERANCE = 1e-9
#: The relative size, against the steady terms, below which a part in the symbol eigenvalue is
#: the round-off of the differences of assemblies, and is zero.
_ROUND_OFF = 1e-12
#: The relative size below which an imaginary part is round-off, and a value is real.
_REAL_TOLERANCE = 1e-10


@dataclass
class EigenmodesResult(ResultTables):
    r"""The outcome of an eigenvalue study: the eigenvalues, closest to
    ``near`` first, and the modes.

    ``near`` is a value of the eigenvalue :math:`\lambda`, except in a vibration
    (``vibration`` true), where it is the angular frequency :math:`\omega` and the
    eigenvalues are closest to :math:`i \omega` first.

    ``modes[k]`` maps each variable to its values in mode ``k``, scaled so
    that the largest value of the mode is 1.  ``summary()`` is the report,
    ``tables["eigenvalues"]`` the table of the eigenvalues, and
    ``write_csv(path)`` and ``write_json(path)`` write them."""

    eigenvalues: numpy.ndarray
    near: float
    num_dofs: int
    residual_norms: numpy.ndarray
    quadratic: bool
    #: A vibration: the only time term is a second time derivative and no term uses
    #: 'eigenvalue', so the eigenvalues are lambda = i omega, with omega the angular frequency
    #: of a mode, and near is an omega.
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
        rows = [
            [k + 1, float(numpy.real(value)), float(numpy.imag(value)), float(norm)]
            for k, (value, norm) in enumerate(zip(self.eigenvalues, self.residual_norms))
        ]
        columns = ["mode", "eigenvalue_real", "eigenvalue_imaginary", "relative_residual"]
        return {"eigenvalues": Table(columns, ["", "", "", ""], rows, title="Eigenvalues")}

    def summary(self) -> str:
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


def _restriction(mask):
    """The indices of the free unknowns, those that ``mask`` does not prescribe, and the matrix
    that restricts a vector of all the unknowns to them."""
    import scipy.sparse as sparse

    index = numpy.flatnonzero(~mask)
    restrict = sparse.csc_matrix(
        (numpy.ones(len(index)), (numpy.arange(len(index)), index)), shape=(len(index), len(mask))
    )
    return index, restrict


def _arnoldi_count(num_modes: int, size: int) -> int:
    """The number of eigenvalues that the Arnoldi iteration computes: more than asked, so that a
    repeated eigenvalue is found with its multiplicity."""
    return min(max(2 * num_modes, num_modes + 5), size - 2)


def _normalised(x):
    """A mode scaled so that its largest value is 1."""
    return x / x[numpy.argmax(numpy.abs(x))]


def _relative_residual(residual, reference: float) -> float:
    return float(numpy.abs(residual).max() / max(reference, _TINY))


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


@dataclass
class _Pencil:
    """The Jacobian J(lambda) = K + lambda D + lambda^2 E on the free unknowns, which are at
    ``index`` among the ``size`` unknowns.  ``symbol_quadratic`` says whether a term in the
    symbol eigenvalue is quadratic in it."""

    K: object
    D: object
    E: object
    index: numpy.ndarray
    size: int
    symbol_quadratic: bool


def _pencil(problem) -> _Pencil:
    """Assembles the Jacobian of the problem at lambda = 0, 1, -1 and 2 and its time terms, and
    checks that it is at most quadratic in lambda."""
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
    # terms, so their round-off is of the size of those.  The time derivatives are assembled
    # alone, and exactly zero where there are none.  Each is measured in its own units.
    steady = max(_norm(stiffness), _norm(plus), _norm(minus), _norm(two), _TINY)
    if _norm(two - (stiffness + 2.0 * linear + 4.0 * quadratic)) > _QUADRATIC_TOLERANCE * steady:
        raise ValueError(
            "solve_modes: the equations depend on the eigenvalue more than quadratically. Write the terms that use 'eigenvalue' as at most quadratic in it."
        )
    if _norm(linear) <= _ROUND_OFF * steady:
        linear = 0.0 * linear
    if _norm(quadratic) <= _ROUND_OFF * steady:
        quadratic = 0.0 * quadratic
    index, restrict = _restriction(mask)
    return _Pencil(
        K=(restrict @ stiffness @ restrict.T).tocsc(),
        D=(restrict @ (linear - mass) @ restrict.T).tocsc(),
        E=(restrict @ (quadratic + inertia) @ restrict.T).tocsc(),
        index=index,
        size=len(mask),
        symbol_quadratic=_norm(quadratic) != 0.0,
    )


@dataclass
class _Eigenproblem:
    """The linear eigenproblem A x = theta B x that gives the eigenvalues of a pencil: theta is
    lambda / time_scale, or mu = lambda^2 for an undamped pencil."""

    A: object
    B: object
    quadratic: bool
    undamped: bool
    vibration: bool
    time_scale: float = 1.0


def _eigenproblem(pencil: _Pencil) -> _Eigenproblem:
    """The linear form of the pencil: the pencil itself when it is linear in lambda, a problem in
    mu = lambda^2 when it has no first-order term, and its companion form otherwise."""
    import scipy.sparse as sparse

    K, D, E = pencil.K, pencil.D, pencil.E
    is_quadratic = _norm(E) > 0.0
    if not is_quadratic and _norm(D) == 0.0:
        raise ValueError(
            "solve_modes: the problem has no time derivative and no term that uses 'eigenvalue', so it defines no eigenvalue."
        )
    undamped = is_quadratic and _norm(D) == 0.0
    # A vibration: the second time derivative is the only time term and no term uses
    # 'eigenvalue', so lambda = i omega with omega the angular frequency, and near is an omega.
    # Any other problem, a quadratic term in the symbol 'eigenvalue' among them, takes near as a
    # value of lambda itself.
    vibration = undamped and not pencil.symbol_quadratic
    kind = {"quadratic": is_quadratic, "undamped": undamped, "vibration": vibration}
    if undamped:
        # K x + lambda^2 E x = 0: a problem in mu = lambda^2 of the size of K, K x = mu (-E) x.
        return _Eigenproblem(K, (-E).tocsc(), **kind)
    if not is_quadratic:
        # K x + lambda D x = 0, that is K x = lambda C x with C = -D.
        return _Eigenproblem(K, (-D).tocsc(), **kind)
    # Companion form: with y = lambda x, [0 I; -K -D] [x; y] = lambda [I 0; 0 E] [x; y].  It is
    # solved in the time scale s = sqrt(|K| / |E|) of the problem, lambda = s lambda~, with K,
    # s D and s^2 E divided by |K|: its blocks are then of one size in any units (in SI a steel
    # structure has |K| / |E| near 1e11, and the unscaled form lost six digits of the damping).
    size_k = max(_norm(K), _TINY)
    time_scale = numpy.sqrt(size_k / _norm(E))
    identity = sparse.identity(K.shape[0], format="csc")
    A = sparse.bmat([[None, identity], [-K / size_k, -D * (time_scale / size_k)]], format="csc")
    B = sparse.bmat([[identity, None], [None, E * (time_scale**2 / size_k)]], format="csc")
    return _Eigenproblem(A, B, **kind, time_scale=time_scale)


def _eigenpairs(form: _Eigenproblem, num_modes: int, near: float, tolerance: float):
    """The num_modes eigenvalues lambda closest to near, closest first, by shift-invert Arnoldi
    iteration, and their eigenvectors."""
    import scipy.sparse.linalg as linalg

    A, B, time_scale = form.A, form.B, form.time_scale
    size = A.shape[0]
    count = _arnoldi_count(num_modes, size)
    # The undamped problem is solved for mu = lambda^2. A vibration has mu = -omega^2, so the
    # shift is -near^2. Another undamped problem has the shift near^2.
    if form.undamped:
        shift = -(near**2) if form.vibration else near**2
    else:
        shift = near / time_scale
    factor = linalg.splu((A - shift * B).tocsc())
    shifted = linalg.LinearOperator((size, size), matvec=lambda v: factor.solve(B @ v), dtype=float)
    mu, vectors = linalg.eigs(shifted, k=count, which="LM", tol=tolerance)
    eigenvalues = (shift + 1.0 / mu) * time_scale
    # The value that the eigenvalues are closest to: i near for a vibration, near otherwise.
    target = 1j * near if form.vibration else near
    if form.undamped:
        # Each mu gives the pair of eigenvalues lambda = +sqrt(mu) and -sqrt(mu), with one mode,
        # and one root is kept. A vibration keeps the root of non-negative imaginary part, i omega
        # (a vibration e^(i omega t)). Another problem keeps the root closer to near, the
        # principal root (of non-negative real part) when both are as close.
        roots = numpy.sqrt(eigenvalues.astype(complex))
        if form.vibration:
            flip = numpy.imag(roots) < 0
        else:
            flip = numpy.abs(-roots - near) < numpy.abs(roots - near)
        eigenvalues = numpy.where(flip, -roots, roots)
    order = numpy.argsort(numpy.abs(eigenvalues - target))[:num_modes]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if not numpy.iscomplexobj(eigenvalues) or numpy.all(
        numpy.abs(numpy.imag(eigenvalues)) <= _REAL_TOLERANCE * numpy.abs(eigenvalues)
    ):
        eigenvalues = numpy.real(eigenvalues)
    return eigenvalues, vectors


def _mode_residual(pencil: _Pencil, value, x) -> float:
    """The relative residual of the mode x of the eigenvalue value: the largest value of
    J(value) x over the largest values of its three terms."""
    K, D, E = pencil.K, pencil.D, pencil.E
    residual = (K + value * D + value**2 * E) @ x
    reference = (
        numpy.abs(K @ x).max()
        + abs(value) * numpy.abs(D @ x).max()
        + abs(value) ** 2 * numpy.abs(E @ x).max()
    )
    return _relative_residual(residual, reference)


def solve_modes(
    problem, num_modes: int = 1, near: float = 0.0, tolerance: float = 1.0e-10
) -> EigenmodesResult:
    """See :meth:`dualmesh.Problem.solve_eigenvalue`."""
    if problem.is_distributed:
        raise ValueError("solve_modes: the eigenvalue study runs on one process at present.")
    if num_modes < 1:
        raise ValueError("solve_modes: num_modes must be at least 1.")
    problem.initialize()
    pencil = _pencil(problem)
    form = _eigenproblem(pencil)
    n = pencil.K.shape[0]
    if num_modes > form.A.shape[0] - 2:
        raise ValueError(
            f"solve_modes: num_modes is {num_modes}, and the problem has {n} free unknowns. Ask for fewer modes."
        )
    eigenvalues, vectors = _eigenpairs(form, num_modes, near, tolerance)
    names = list(problem._problem.variable_names())
    modes, residuals = [], []
    for k in range(len(eigenvalues)):
        x = vectors[:n, k]
        if numpy.all(numpy.abs(numpy.imag(x)) <= _REAL_TOLERANCE * numpy.abs(x).max()):
            x = numpy.real(x)
        x = _normalised(x)
        residuals.append(_mode_residual(pencil, eigenvalues[k], x))
        modes.append(_mode_values(problem, pencil.index, pencil.size, x, names))
    # The solution holds the real part of the first mode.
    first = numpy.zeros(pencil.size)
    first[pencil.index] = numpy.real(_normalised(vectors[:n, 0]))
    problem._problem.set_solution(list(first))
    return EigenmodesResult(
        eigenvalues=numpy.asarray(eigenvalues),
        near=float(near),
        num_dofs=n,
        residual_norms=numpy.asarray(residuals),
        quadratic=bool(form.quadratic),
        vibration=bool(form.vibration),
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

    def to_dict(self) -> dict:
        return {
            "load_factors": [float(v) for v in self.load_factors],
            "residual_norms": [float(v) for v in self.residual_norms],
            "modes": [
                {name: [float(x) for x in values] for name, values in mode.items()}
                for mode in self.modes
            ],
        }

    def __str__(self) -> str:
        return self.summary()


def solve_buckling(problem, num_modes: int = 1, tolerance: float = 1.0e-10) -> BucklingResult:
    """See :meth:`dualmesh.Problem.solve_buckling`."""
    import scipy.sparse.linalg as linalg

    if num_modes < 1:
        raise ValueError("solve_buckling: num_modes must be at least 1.")
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
    index, restrict = _restriction(mask)
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
    count = _arnoldi_count(num_modes, n)
    inverse, vectors = linalg.eigs(operator, k=count, which="LR", tol=tolerance)
    # The unknowns on which K_sigma does not act have no load factor: round-off leaves their
    # values 1/lambda at about machine precision times the spectral radius of the operator, of
    # either sign.  Only a value above that is a load factor.
    radius = numpy.abs(
        linalg.eigs(operator, k=1, which="LM", tol=1e-6, return_eigenvectors=False)
    ).max()
    keep = numpy.real(inverse) > 1e-10 * radius
    found = int(numpy.count_nonzero(keep))
    if found < num_modes:
        raise ValueError(
            f"solve_buckling: the solved state has {found} positive load factors, and num_modes asks for {num_modes}. Only a state under compression buckles at a positive multiple of its load, so solve a compressive load first, or ask for fewer modes."
        )
    factors = 1.0 / numpy.real(inverse[keep])
    vectors = numpy.real(vectors[:, keep])
    order = numpy.argsort(factors)[:num_modes]
    factors, vectors = factors[order], vectors[:, order]
    names = list(problem._problem.variable_names())
    modes, residuals = [], []
    for k in range(len(factors)):
        x = _normalised(vectors[:, k])
        residual = (K + factors[k] * G) @ x
        reference = numpy.abs(K @ x).max() + abs(factors[k]) * numpy.abs(G @ x).max()
        residuals.append(_relative_residual(residual, reference))
        modes.append(_mode_values(problem, index, len(mask), x, names))
    problem._problem.set_solution(list(solved))
    return BucklingResult(
        load_factors=numpy.asarray(factors), residual_norms=numpy.asarray(residuals), modes=modes
    )
