.. _theory-eigenvalues:

Eigenvalue studies
==================

:meth:`~dualmesh.Problem.solve_eigenvalue` computes eigenvalues and modes of
any problem: the modes of decay of a diffusion problem, the modes of
vibration of a structure, or the criticality of a reactor.

The eigenvalue
--------------

Every field varies in time as

.. math::

   u(t) = \hat{u}\, e^{-\lambda t} ,

so a time derivative :math:`d\, \partial u / \partial t` becomes
:math:`-\lambda d\, \hat{u}`.  The expressions may also use the symbol
``eigenvalue`` for :math:`\lambda`, for example in a source.  The study
linearizes the equations at the current solution.  Their Jacobian at
:math:`\lambda` is

.. math::
   :label: eigen-jacobian

   \mathbf{J}(\lambda) = \mathbf{K} + \lambda (\mathbf{B}_1 - \mathbf{M}) + \lambda^2 \mathbf{B}_2 ,

where :math:`\mathbf{K}` is the Jacobian of the steady terms at
:math:`\lambda = 0`, :math:`\mathbf{M}` the Jacobian of the time derivatives,
and :math:`\mathbf{B}_1` and :math:`\mathbf{B}_2` are the linear and the
quadratic parts of the terms that use ``eigenvalue``.  Three assemblies, at
:math:`\lambda = 0`, 1 and -1, give :math:`\mathbf{B}_1` and
:math:`\mathbf{B}_2` exactly.  A fourth assembly, at :math:`\lambda = 2`,
makes sure that the terms are at most quadratic in :math:`\lambda`.

The eigenvalues are the values of :math:`\lambda` for which
:math:`\mathbf{J}(\lambda) \hat{\mathbf{u}} = 0` has a solution that is not
0.  The prescribed values of the boundary conditions are set to 0, and a
source that does not depend on the fields has no effect, because the
Jacobian does not contain it.

The solution
------------

With :math:`\mathbf{B}_2 = 0` the problem is
:math:`\mathbf{K} \hat{\mathbf{u}} = \lambda \mathbf{C} \hat{\mathbf{u}}`
with :math:`\mathbf{C} = \mathbf{M} - \mathbf{B}_1`.  The study computes the
eigenvalues closest to ``near`` by the Arnoldi method with implicit restarts
(ARPACK through SciPy) applied to the shifted and inverted operator
:math:`(\mathbf{K} - \sigma \mathbf{C})^{-1} \mathbf{C}`, where
:math:`\sigma` is ``near``.  An eigenvalue :math:`\mu` of this operator gives
:math:`\lambda = \sigma + 1/\mu`.  A singular :math:`\mathbf{C}`, for example
of a constraint, does not stop the method.  A quadratic problem is solved in
its companion form, a linear problem of twice the size for :math:`\hat{\mathbf{u}}`
and :math:`\lambda \hat{\mathbf{u}}`.

The study computes twice as many eigenvalues as asked, and at least five
more, so that a repeated eigenvalue is found with its multiplicity.  It
returns the ``num_modes`` eigenvalues closest to ``near`` in an
:class:`~dualmesh.modes.EigenmodesResult`, with their modes scaled to a
largest value of 1, and leaves the real part of the first mode in the
solution.  An eigenvalue can be complex, and the result gives its real and
imaginary parts.

Verification
------------

The tests in ``tests/python/test_eigenvalue_study.py`` compare the study
with exact eigenvalues:

* The Laplacian on the unit square with zero values on the boundary has the
  eigenvalues :math:`\pi^2 (m^2 + n^2)`.  All four methods converge to the
  first four at second order in the element size, and the double eigenvalue
  :math:`5\pi^2` is found twice.
* The same eigenvalues come from the time derivative and from the symbol
  ``eigenvalue``, to round-off, and a quadratic symbol gives their square
  roots.
* The operator :math:`-u'' + c u'` on :math:`(0, 1)` is not symmetric.  Its
  eigenvalues are :math:`(n\pi)^2 + c^2/4`, and all four methods converge to
  them at second order.
* Two diffusion equations coupled by a rotation of rate :math:`\omega` have
  the complex eigenvalues :math:`\mu \pm i \omega`, where :math:`\mu` is an
  eigenvalue of the Laplacian.

:ref:`tutorial-criticality` applies the study to the criticality of a
reactor.
