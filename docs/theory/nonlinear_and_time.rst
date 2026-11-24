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
``ctx.coefficient_value(variable)``.  In Newton mode the same call returns the
current AD value, so one kernel serves both schemes.  With
``nonlinear_solver="Picard"`` the iteration is direct iteration, and its
acceleration (relaxation) parameter :math:`\gamma`,

.. math::

   \bar{U} = (1-\gamma)\,U^{r} + \gamma\, U^{r-1} , \qquad 0 \le \gamma < 1 ,

is the ``relaxation`` option.  The von Kármán beams of the verification suite
use :math:`\gamma = 0.35`, and the lid-driven cavity at :math:`Re = 1000` converges
with :math:`\gamma = 0.5`.

**Continuation.**  When Newton's method does not converge on a steady problem
itself, the solve takes it again from the starting state, with the data that
drive it ramped by a fraction :math:`\lambda` from 0 to 1: the sources, body
forces and buoyancy, prescribed fluxes, tractions and pressures, point loads and
the distributed loads of plates and beams are multiplied by :math:`\lambda`
(``Object::isLoad`` in C++), and each prescribed value moves from its starting
value :math:`u_0` to :math:`u_0 + \lambda (g - u_0)`, so that a wall at 300 K
is never taken through 0 K.  The converged solution of one step is the initial
guess of the next.  The first step is :math:`\lambda = 1/2`; a step that
converges doubles the next one, and a step that fails is halved and taken again
from the last converged state, down to :math:`1/1024`.  COMSOL's stationary
solver continues the same way when its damped Newton iteration fails.  A
problem that nothing can ramp (a solution-dependent source alone) is not
continued.  On a :math:`32 \times 32` mesh the lid-driven cavity takes 3 steps
at :math:`Re = 1000` and 5 at :math:`Re = 3200`, and a bar whose conductivity
grows as :math:`T^6`, heated from 300 K to 3000 K, takes 12.  The steps of a
transient run are not continued: a step that fails is retried with a smaller
time step instead.

Convergence is tested per equation, so that the tests have no unit.  The
residual of equation :math:`v` (the unknowns of one variable) is measured
against its **reference residual**, the size of the terms of that equation:
the assembly adds the magnitude of the contribution of each element (each
boundary condition, each point source) to each unknown,

.. math::

   R^{\text{ref}}_i = \sum_e \lvert R^e_i \rvert ,

which is zero at the prescribed values.  Terms that cancel inside one element
(a body force against a pressure gradient) add their net value, so the
reference is the size of what the elements exchange.  An equation whose terms
all vanish (the momentum of a uniform flow) has no size of its own, and it has
converged at the round-off level of its rows,
:math:`64\,\varepsilon\, \lVert (|J|\,|U|)_v \rVert`.  Its norm over the unknowns of a
variable has the units of that equation (the conducted and carried heat, the
forces), does not depend on the start of the step, and is not zero where the
reactions are.  The iteration has converged when

.. math::

   \lVert R_v \rVert \le \varepsilon_r\, \lVert R^{\text{ref}}_v \rVert
   \quad \text{for every equation } v ,

with :math:`\varepsilon_r` the ``relative_tolerance``.  A small step does not
count as convergence: an iteration that stalls with a residual above the
tolerance has not found the solution.

The line search, the divergence test and the printed residual use the scaled
norm :math:`\bigl(\sum_v (\lVert R_v \rVert / s_v)^2\bigr)^{1/2}`, with
:math:`s_v` the largest reference norm of equation :math:`v` in the solve.
This is the reference residual of MOOSE's ``ReferenceResidualProblem``, with
the magnitudes of the contributions as the reference and no input.  OpenFOAM
normalises the residual of each equation by its own matrix, and COMSOL scales
the variables and the equations, for the same reason: a norm that adds the
residuals of several equations is decided by the equation with the largest
numbers, which depends on the units.

The iterative linear solvers measure the residual in the same way: GMRES runs
on :math:`W J\, \delta U = -W R`, with :math:`W` one over the size of the terms
of each equation, and applies the block preconditioners of a flow, whose
operators come from the physics, as :math:`P^{-1} W^{-1}`; ILU and algebraic
multigrid are built from :math:`W J`.  The row of a prescribed value takes the
mean size of the diagonal of its equation instead of 1.  A change of units
multiplies each equation and each unknown by a constant, which these
operations absorb, so the iterations are the same in any units
(``tests/python/test_units.py``).  The direct solver needs none of this.  A
symmetric matrix keeps its symmetry, which CG needs, and PETSc with a block
split takes the system as it is.

Time integration
----------------

Transient problems use the :math:`\theta` method.  With
:math:`R_{\text{time}}` the residual of the time-derivative kernels and
:math:`R_{\text{ss}}` the rest,

.. math::

   R_{\text{time}}(U^{n+1}) + \theta\, R_{\text{ss}}(U^{n+1}, t^{n+1})
   + (1-\theta)\, R_{\text{ss}}(U^{n}, t^{n}) = 0 ,

so :math:`\theta = 1` is the backward Euler method, :math:`\theta = 1/2` the
Crank-Nicolson method, and :math:`\theta = 0` the forward Euler method.  A
time-derivative kernel with ``quadrature="nodal"`` gives the lumped capacity
matrix, which in the dual mesh method is the measure of the control domain
times the nodal rate.

**Second-order equations.**  An equation with a second time derivative (the
inertia :math:`\rho\, \ddot{u}` of a solid, or the ``m`` of
``general_form_PDE``) is integrated by the generalized-:math:`\alpha` method of
Chung and Hulbert [ChungHulbert1993]_.  With the velocity :math:`v` and the
acceleration :math:`a` of each unknown kept from step to step, the Newmark
relations

.. math::

   u^{n+1} = u^n + \Delta t\, v^n + \Delta t^2 \bigl((\tfrac{1}{2} - \beta) a^n
   + \beta a^{n+1}\bigr), \qquad
   v^{n+1} = v^n + \Delta t \bigl((1 - \gamma) a^n + \gamma a^{n+1}\bigr)

give :math:`a^{n+1}` and :math:`v^{n+1}` as functions of :math:`u^{n+1}`, and the
equation is taken at intermediate levels:

.. math::

   M a^{n+1-\alpha_m} + C v^{n+1-\alpha_f} + R_{\text{ss}}(u^{n+1-\alpha_f}) = 0,
   \qquad x^{n+1-\alpha} = (1-\alpha)\, x^{n+1} + \alpha\, x^n .

The steady terms are weighted as the :math:`\theta` method weights them, with
:math:`\theta = 1 - \alpha_f`, so the first-order equations of the same problem
take that :math:`\theta`.  The parameters follow from the spectral radius
:math:`\rho_\infty` at infinite frequency (``spectral_radius``, default 0.75,
COMSOL's default): :math:`\alpha_m = (2\rho_\infty - 1)/(\rho_\infty + 1)`,
:math:`\alpha_f = \rho_\infty/(\rho_\infty + 1)`, :math:`\gamma = 1/2 -
\alpha_m + \alpha_f` and :math:`\beta = (1 - \alpha_m + \alpha_f)^2/4`.  The
method is second order and unconditionally stable, and damps the modes the
step does not resolve by :math:`\rho_\infty` per step: the amplification matrix
has the triple root :math:`-\rho_\infty` at infinite frequency, which the tests
measure (0.5001 and 0.7548 for 0.5 and 0.75).  ``time_scheme="Newmark"`` is
:math:`\alpha_m = \alpha_f = 0`, :math:`\beta = 1/4`, :math:`\gamma = 1/2`,
the average acceleration rule, which keeps the energy of a linear undamped
system to round-off.  The acceleration of the start solves
:math:`M a^0 = -R(u^0, v^0)`, and a prescribed value that changes in time sets
its own velocity and acceleration.

**Natural frequencies.**  The eigenvalue study takes the second time derivatives
as the quadratic term in :math:`\lambda` of :math:`u = \hat{u} e^{-\lambda t}`.
Without damping it solves :math:`K x = \omega^2 M x`; with damping it solves the
companion form, in the time scale :math:`\sqrt{\lVert K \rVert / \lVert M
\rVert}` of the problem, so that its blocks are of one size in any units.
``frequencies`` gives :math:`|\operatorname{Im} \lambda| / 2\pi`.

Solvers and complexity
----------------------

PETSc [PETSc2023]_ solves the linear systems.  By default it factorises a
system where that is cheap, and otherwise iterates with a preconditioner that
suits the structure of the system.  :ref:`The choice of linear solver
<linear-solver-choice>` gives the choice and its reasons.  The unknowns are
ordered node by node (``dof = node * num_variables + variable``),
which keeps the couplings of a multi-field model close to the diagonal.
