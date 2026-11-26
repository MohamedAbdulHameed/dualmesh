.. _theory-eigenvalues:

Eigenvalue studies
==================

:meth:`~dualmesh.Problem.solve_eigenvalue` computes eigenvalues and modes of a
problem: the modes of decay of a diffusion problem, the modes of vibration of a
structure, or the criticality of a reactor.  This page gives the eigenvalue
problem that the study forms and the method that solves it.

.. contents::
   :local:
   :depth: 1

The eigenvalue problem
----------------------

The study takes every field to vary in time as

.. math::

   u(\mathbf{x}, t) = \hat{u}(\mathbf{x}) \, e^{-\Lambda t} ,

with :math:`\hat{u}` the mode, :math:`t` the time (s) and :math:`\Lambda` the
eigenvalue, in 1/s when the time derivatives define it.  A first time derivative
:math:`c \, \partial u / \partial t`, with :math:`c` the capacity, becomes
:math:`-\Lambda c \, \hat{u}`, and a second time derivative
:math:`c \, \partial^2 u / \partial t^2` becomes :math:`\Lambda^2 c \, \hat{u}`.
The expressions of a problem may also use the symbol ``eigenvalue`` for
:math:`\Lambda`, for example in a source, and the unit of :math:`\Lambda` then
follows from the terms that use it.

The study linearizes the equations at the current solution.  With
:math:`\hat{\mathbf{u}}` the vector of the nodal values of the mode, their
Jacobian at :math:`\Lambda` is

.. math::
   :label: eigen-jacobian

   \mathbf{J}(\Lambda) = \mathbf{K} + \Lambda (\mathbf{B}_1 - \mathbf{M}_1)
     + \Lambda^2 (\mathbf{B}_2 + \mathbf{M}_2) ,

with these matrices:

* :math:`\mathbf{K}`, the Jacobian of the steady terms at :math:`\Lambda = 0`,
* :math:`\mathbf{M}_1` and :math:`\mathbf{M}_2`, the Jacobians of the first and
  the second time derivatives, with the consistent storage (the Gauss rule)
  where time stepping lumps it,
* :math:`\mathbf{B}_1` and :math:`\mathbf{B}_2`, the linear and the quadratic
  parts of the terms that use ``eigenvalue``.

Three assemblies of the steady terms, at :math:`\Lambda = 0`, 1 and -1, give
:math:`\mathbf{B}_1` and :math:`\mathbf{B}_2` exactly, and a fourth, at
:math:`\Lambda = 2`, checks that the terms are at most quadratic in
:math:`\Lambda`.  The study refuses a problem with a stronger dependence on
:math:`\Lambda`, and a problem with neither a time derivative nor the symbol
``eigenvalue``.

The eigenvalues are the values of :math:`\Lambda` for which
:math:`\mathbf{J}(\Lambda) \hat{\mathbf{u}} = 0` has a nonzero solution.  The
unknowns with a prescribed value are left out, which sets the prescribed values
of the mode to zero.  A source that does not depend on the fields has no effect,
because the Jacobian does not hold it.

The solution
------------

The study computes the eigenvalues closest to the value :math:`\Lambda_0` that
``near`` gives, by the Arnoldi method with implicit restarts (ARPACK through
SciPy) applied to a shifted and inverted operator.  It takes one of three forms.

**Linear in** :math:`\Lambda`.  With :math:`\mathbf{B}_2 + \mathbf{M}_2 = 0`,
the problem is
:math:`\mathbf{K} \hat{\mathbf{u}} = \Lambda \mathbf{C} \hat{\mathbf{u}}` with
:math:`\mathbf{C} = \mathbf{M}_1 - \mathbf{B}_1`.  The Arnoldi method runs on
:math:`(\mathbf{K} - \Lambda_0 \mathbf{C})^{-1} \mathbf{C}`, and an eigenvalue
:math:`z` of this operator gives :math:`\Lambda = \Lambda_0 + 1/z`.  A
singular :math:`\mathbf{C}`, such as that of a constraint, does not stop the
method, because only :math:`\mathbf{K} - \Lambda_0 \mathbf{C}` is factorized.

**Quadratic without a linear term.**  With
:math:`\mathbf{B}_1 - \mathbf{M}_1 = 0`, as for an undamped structure, the
problem :math:`\mathbf{K} \hat{\mathbf{u}} = -\Lambda^2 (\mathbf{B}_2 +
\mathbf{M}_2) \hat{\mathbf{u}}` is linear in :math:`\Lambda^2`, with the shift
:math:`\Lambda_0^2`.  The study takes the root :math:`\Lambda` with a non-negative
imaginary part.  For a structure with the stiffness :math:`\mathbf{K}` and the
mass :math:`\mathbf{M}_2`, the vibration :math:`\Lambda = i\omega` gives
:math:`\mathbf{K} \hat{\mathbf{u}} = \omega^2 \mathbf{M}_2 \hat{\mathbf{u}}`,
with :math:`\omega` the angular frequency (rad/s).

**Quadratic.**  With both terms, the study solves the companion form, a linear
problem of twice the size for :math:`\hat{\mathbf{u}}` and
:math:`\Lambda \hat{\mathbf{u}}`.  It writes the form in the rate scale
:math:`\Lambda_s = (\lVert \mathbf{K} \rVert / \lVert \mathbf{B}_2 + \mathbf{M}_2
\rVert)^{1/2}` (1/s) of the problem, with :math:`\lVert \cdot \rVert` the largest
magnitude of an entry, and divides the blocks by :math:`\lVert \mathbf{K}
\rVert`, so the blocks are of one size in any units.  With
:math:`\Lambda = \Lambda_s \tilde{\Lambda}`, :math:`\mathbf{E} = \mathbf{B}_2 +
\mathbf{M}_2` and :math:`\mathbf{y} = \tilde{\Lambda} \hat{\mathbf{u}}`, the
problem is

.. math::

   \begin{bmatrix} \mathbf{0} & \mathbf{I} \\
     -\mathbf{K} / \lVert \mathbf{K} \rVert &
     -\Lambda_s (\mathbf{B}_1 - \mathbf{M}_1) / \lVert \mathbf{K} \rVert
   \end{bmatrix}
   \begin{bmatrix} \hat{\mathbf{u}} \\ \mathbf{y} \end{bmatrix}
   = \tilde{\Lambda}
   \begin{bmatrix} \mathbf{I} & \mathbf{0} \\
     \mathbf{0} & \Lambda_s^2 \mathbf{E} / \lVert \mathbf{K} \rVert
   \end{bmatrix}
   \begin{bmatrix} \hat{\mathbf{u}} \\ \mathbf{y} \end{bmatrix} ,

with :math:`\mathbf{I}` the identity, and the shift is :math:`\Lambda_0 / \Lambda_s`.

The result
----------

The study computes twice as many eigenvalues as asked, and at least five more,
so it finds a repeated eigenvalue with its multiplicity.  It returns the
``num_modes`` eigenvalues closest to ``near`` in an
:class:`~dualmesh.modes.EigenmodesResult`, with each mode scaled so that its
entry of largest magnitude is 1, and the relative residual
:math:`\lVert \mathbf{J}(\Lambda) \hat{\mathbf{u}} \rVert` of each pair over the
sizes of its terms.  It leaves the real part of the first mode in the solution.
An eigenvalue can be complex, and the result gives its real and imaginary
parts.  ``frequencies`` gives :math:`|\operatorname{Im} \Lambda| / (2\pi)` in
Hz.  The study runs on one process.

:ref:`tutorial-criticality` applies the study to the criticality of a reactor,
and :doc:`/verification` gives the cases that check it.
