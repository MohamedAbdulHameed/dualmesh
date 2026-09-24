Nonlinear problems, time integration and solvers
================================================

The discretisations of the previous chapters turn a boundary value problem into
a system of algebraic equations.  This chapter is about solving that system:
how a nonlinear system is linearised and iterated, how a time-dependent problem
is stepped, and what the linear solvers cost.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Nonlinear problems
------------------

Two schemes are available, both driven by the same residual.

**Newton's method** uses the exact Jacobian.  Every kernel is written in terms
of the forward-mode automatic differentiation type :class:`ADReal`, whose
partial derivatives with respect to the local (element) degrees of freedom are
carried through every arithmetic operation.  The element contributions to
:math:`\partial R_I / \partial U_J` therefore need no hand coding and are exact
for any nonlinearity, including nonlinear boundary conditions.

**Direct (Picard) iteration** freezes the nonlinear coefficients at the
previous iterate.  A kernel asks for the lagged value with
``ctx.coefficient_value(variable)`` — in Newton mode the same call returns the
current AD value, so one kernel serves both schemes.  With
``nonlinear_solver="picard"`` the iteration is that of Section 6.2 of the book,
and the acceleration (relaxation) parameter of Eq. (6.2.15),

.. math::

   \bar{U} = (1-\gamma)\,U^{r} + \gamma\, U^{r-1} , \qquad 0 \le \gamma < 1 ,

is the ``relaxation`` option.  The nonlinear beam problems of Section 7.6 use
:math:`\gamma = 0.35`, and the lid-driven cavity at :math:`Re = 1000` converges
with :math:`\gamma = 0.5`.

**Load stepping** applies the loads in increments, taking the converged
solution of one step as the initial guess of the next.  Objects whose
contribution scales with the load (body forces, tractions, distributed loads,
point loads) are multiplied by the load factor; a Dirichlet condition can be
ramped too by setting ``scale_with_load=True``.  The lid-driven cavity at
:math:`Re = 1000` does not converge from rest with Newton's method but does
converge in a handful of load steps, and the nonlinear beams of Table 7.6.1 use
increments of :math:`\Delta q_0 = 1`.

Convergence is declared when the residual norm drops below
``absolute_tolerance``, or below ``relative_tolerance`` times its initial
value, or when the relative solution increment

.. math::

   \frac{\lVert U^{r+1} - U^{r}\rVert}{\lVert U^{r+1}\rVert} \le \varepsilon

falls below ``step_tolerance``, which is the criterion used in the book.

Time integration
----------------

Transient problems use the :math:`\theta` method.  With
:math:`R_{\text{time}}` the residual of the time-derivative kernels and
:math:`R_{\text{ss}}` the rest,

.. math::

   R_{\text{time}}(U^{n+1}) + \theta\, R_{\text{ss}}(U^{n+1}, t^{n+1})
   + (1-\theta)\, R_{\text{ss}}(U^{n}, t^{n}) = 0 ,

so :math:`\theta = 1` is the backward Euler method, :math:`\theta = 1/2` the
Crank–Nicolson method, and :math:`\theta = 0` the forward Euler method.  A
time-derivative kernel with ``quadrature="nodal"`` gives the lumped capacity
matrix, which in the dual mesh method is simply the measure of the control
domain times the nodal rate.

Solvers and complexity
----------------------

The linear systems are solved with Eigen [Eigen]_: by default
(``linear_solver="automatic"``) with a sparse LU factorisation where that is
cheap and with BiCGSTAB preconditioned by an incomplete LU factorisation
without fill where it is not, the choice and its reasons being given in
:ref:`the choice of linear solver <linear-solver-choice>`; GMRES and the
conjugate gradient method (for symmetric problems) are also available.  The unknowns are
ordered node by node (``dof = node * num_variables + variable``), which keeps
the couplings of a multi-field model close to the diagonal.
