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

import numpy as np

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

    eigenvalues: np.ndarray
    near: float
    num_dofs: int
    residual_norms: np.ndarray
    quadratic: bool
    modes: list = field(default_factory=list, repr=False)

    @property
    def tables(self) -> dict:
        from .tables import Table

        rows = [
            [k + 1, float(np.real(value)), float(np.imag(value)), float(norm)]
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
            "eigenvalues_real": [float(np.real(v)) for v in self.eigenvalues],
            "eigenvalues_imaginary": [float(np.imag(v)) for v in self.eigenvalues],
            "relative_residuals": [float(r) for r in self.residual_norms],
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)


def _operator(problem, steady_terms: bool, time_kernels: bool, eigenvalue: float):
    import scipy.sparse as sparse

    values, indices, indptr, mask = problem._problem._operator(
        steady_terms, time_kernels, eigenvalue
    )
    n = len(indptr) - 1
    return sparse.csc_matrix((values, indices, indptr), shape=(n, n)), np.asarray(mask, dtype=bool)


def _norm(matrix) -> float:
    return float(abs(matrix).max()) if matrix.nnz else 0.0


def _mode_values(problem, index, size, x, names) -> dict:
    """The values of every variable in a mode ``x`` of the free unknowns."""
    parts = {}
    for part, values in (("real", np.real(x)), ("imaginary", np.imag(x))):
        full = np.zeros(size)
        full[index] = values
        problem._problem.set_solution(list(full))
        parts[part] = {name: np.asarray(problem.values(name)) for name in names}
    if not np.iscomplexobj(x):
        return parts["real"]
    return {name: parts["real"][name] + 1j * parts["imaginary"][name] for name in names}


def solve_modes(
    problem, num_modes: int = 1, near: float = 0.0, tolerance: float = 1.0e-10
) -> EigenmodesResult:
    """See :meth:`dualmesh.Problem.solve_modes`."""
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    if problem.is_distributed:
        raise ValueError("solve_modes: the eigenvalue study runs on one process at present.")
    if num_modes < 1:
        raise ValueError("solve_modes: num_modes must be at least 1.")
    problem.initialize()
    stiffness, mask = _operator(problem, True, False, 0.0)
    mass, _ = _operator(problem, False, True, 0.0)
    plus, _ = _operator(problem, True, False, 1.0)
    minus, _ = _operator(problem, True, False, -1.0)
    two, _ = _operator(problem, True, False, 2.0)
    linear = (plus - minus) / 2.0
    quadratic = (plus + minus) / 2.0 - stiffness
    scale = max(_norm(stiffness), _norm(linear), _norm(quadratic), _norm(mass), 1e-300)
    if _norm(two - (stiffness + 2.0 * linear + 4.0 * quadratic)) > 1e-9 * scale:
        raise ValueError(
            "solve_modes: the equations depend on the eigenvalue more than quadratically. Write the terms that use 'eigenvalue' as at most quadratic in it."
        )
    free = ~mask
    index = np.flatnonzero(free)
    restrict = sparse.csc_matrix(
        (np.ones(len(index)), (np.arange(len(index)), index)), shape=(len(index), len(mask))
    )
    K = (restrict @ stiffness @ restrict.T).tocsc()
    D = (restrict @ (linear - mass) @ restrict.T).tocsc()
    E = (restrict @ quadratic @ restrict.T).tocsc()
    n = K.shape[0]
    is_quadratic = _norm(E) > 1e-12 * scale
    if not is_quadratic and _norm(D) == 0.0:
        raise ValueError(
            "solve_modes: the problem has no time derivative and no term that uses 'eigenvalue', so it defines no eigenvalue."
        )
    if is_quadratic:
        # Companion form: with y = lambda x, [0 I; -K -D] [x; y] = lambda [I 0; 0 E] [x; y].
        identity = sparse.identity(n, format="csc")
        A = sparse.bmat([[None, identity], [-K, -D]], format="csc")
        B = sparse.bmat([[identity, None], [None, E]], format="csc")
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
    factor = linalg.splu((A - near * B).tocsc())
    shifted = linalg.LinearOperator((size, size), matvec=lambda v: factor.solve(B @ v), dtype=float)
    mu, vectors = linalg.eigs(shifted, k=count, which="LM", tol=tolerance)
    eigenvalues = near + 1.0 / mu
    order = np.argsort(np.abs(eigenvalues - near))[:num_modes]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if not np.iscomplexobj(eigenvalues) or np.all(
        np.abs(np.imag(eigenvalues)) <= 1e-10 * np.abs(eigenvalues)
    ):
        eigenvalues = np.real(eigenvalues)
    names = list(problem._problem.variable_names())
    modes, residuals = [], []
    for k in range(len(eigenvalues)):
        x = vectors[:n, k]
        if np.all(np.abs(np.imag(x)) <= 1e-10 * np.abs(x).max()):
            x = np.real(x)
        x = x / x[np.argmax(np.abs(x))]
        value = eigenvalues[k]
        residual = (K + value * D + value**2 * E) @ x
        reference = (
            np.abs(K @ x).max()
            + abs(value) * np.abs(D @ x).max()
            + abs(value) ** 2 * np.abs(E @ x).max()
        )
        residuals.append(float(np.abs(residual).max() / max(reference, 1e-300)))
        modes.append(_mode_values(problem, index, len(mask), x, names))
    # The solution holds the real part of the first mode.
    first = np.zeros(len(mask))
    first[index] = np.real(vectors[:n, 0] / vectors[np.argmax(np.abs(vectors[:n, 0])), 0])
    problem._problem.set_solution(list(first))
    return EigenmodesResult(
        eigenvalues=np.asarray(eigenvalues),
        near=float(near),
        num_dofs=n,
        residual_norms=np.asarray(residuals),
        quadratic=bool(is_quadratic),
        modes=modes,
    )
