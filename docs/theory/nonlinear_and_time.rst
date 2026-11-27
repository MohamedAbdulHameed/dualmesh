Nonlinear problems, time integration and solvers
================================================

The discretizations of :doc:`finite_volume` turn a boundary value problem into a
system of algebraic equations, one per unknown.  This page describes how the code
solves that system: the nonlinear iteration and its convergence test, the
continuation of a steady solve, the time integration of first- and second-order
equations, and the control of the time step.

Throughout the page, :math:`\mathbf{U}` is the vector of the nodal unknowns,
:math:`\mathbf{R}(\mathbf{U})` the vector of the discrete equations (the
residual), with one entry per unknown in the units of its equation, and
:math:`\mathbf{J} = \partial \mathbf{R} / \partial \mathbf{U}` the Jacobian
matrix.

.. contents::
   :local:
   :depth: 1

Nonlinear iteration
-------------------

The nonlinear solver of the Portable, Extensible Toolkit for Scientific
Computation (PETSc) [PETSc2023]_, SNES, runs the iteration.  The option
``nonlinear_solver`` chooses its linearization.

``"Newton"`` (the default)
   Each step solves :math:`\mathbf{J} \, \delta\mathbf{U} = -\mathbf{R}` and
   updates :math:`\mathbf{U}` by :math:`\delta\mathbf{U}`.  Every kernel is
   written with a forward-mode automatic differentiation type [Wengert1964]_,
   which carries the derivatives with respect to the unknowns of the element
   through every operation.  The Jacobian is therefore exact for any
   nonlinearity, nonlinear boundary conditions included, without derivatives
   written by hand.

``"Picard"``
   Direct iteration.  The coefficients that a kernel reads as coefficients
   take the value of the previous iterate and carry no derivative, so each step
   solves the equations with frozen coefficients.  The same kernel serves both
   iterations.  An expression, such as a property formula or the flux of
   ``parsed_kernel``, takes the current iterate with its derivatives under
   both iterations, because it can be a term of an equation as well as a
   coefficient.

``"linear"``
   One linear solve.  A residual above the tolerance after it stops the solve
   with an error, because the problem is then nonlinear or the linear solve
   failed.

**Relaxation.**  Without a line search, each step after the first takes the
fraction :math:`1 - \gamma_\mathrm{r}` of its increment,

.. math::

   \mathbf{U}^{r+1} = \mathbf{U}^{r} + (1 - \gamma_\mathrm{r}) \,
   \delta\mathbf{U}^{r} ,
   \qquad 0 \le \gamma_\mathrm{r} < 1 ,

with :math:`r` the iteration number and :math:`\gamma_\mathrm{r}` the
``relaxation`` (default 0).

**Line search.**  With ``line_search="backtracking"``, Newton's method halves
the step length :math:`\alpha_\mathrm{ls}` from 1 until the scaled residual norm of the next
section satisfies
:math:`\lVert \mathbf{R}(\mathbf{U} + \alpha_\mathrm{ls} \, \delta\mathbf{U}) \rVert_s \le
(1 - 10^{-4} \alpha_\mathrm{ls}) \lVert \mathbf{R}(\mathbf{U}) \rVert_s`, at most
``max_line_search_steps`` times (default 10).  The default ``"automatic"`` turns
the line search on when a kernel uses the limited advection scheme, whose
limiter has kinks on which the full step can cycle, and off otherwise.

The convergence test
--------------------

The code tests each equation against the size of its own terms, so the test has
no unit.  An equation :math:`v` is the set of the entries of :math:`\mathbf{R}`
that belong to one variable.  Its reference residual adds the magnitude of the
contribution of each element, boundary condition and point load to each entry
:math:`i`,

.. math::

   R^{\mathrm{ref}}_i = \sum_e \lvert R^e_i \rvert ,

with :math:`R^e_i` the contribution of the object :math:`e`.  It is zero at the
prescribed values.  Terms that cancel inside one element, such as a body force
against a pressure gradient, add their net value, so the reference measures
what the elements exchange.  The iteration has converged when, for every
equation :math:`v`,

.. math::

   \lVert \mathbf{R}_v \rVert \le \max\Bigl(
      \epsilon_\mathrm{rel} \, \lVert \mathbf{R}^{\mathrm{ref}}_v \rVert ,\quad
      64 \, \epsilon_\mathrm{mach} \, \lVert \mathbf{L}_v \rVert \Bigr) ,

with :math:`\lVert \cdot \rVert` the Euclidean norm over the entries of
:math:`v`, :math:`\epsilon_\mathrm{rel}` the ``relative_tolerance`` (default
:math:`10^{-10}`) and :math:`\epsilon_\mathrm{mach}` the machine precision.  The
round-off level :math:`\mathbf{L} = |\mathbf{J}| \, |\mathbf{U}| + \mathbf{R}^\mathrm{term}`
adds the product of the magnitudes of the Jacobian and of the unknowns, a matrix
times a vector, to
the size :math:`\mathbf{R}^\mathrm{term}` of the terms inside the fluxes and the sources.  It
is the floor for an equation whose terms all vanish, such as the momentum of a
uniform flow, or cancel, such as the stress of a stiff solid at a small strain.
The components of a vector unknown, a velocity or a displacement, take the test
as one equation, with the norms over all their entries: the share of one
component depends on the orientation of the axes, and a component whose terms
all vanish, such as the transverse momentum of a plug flow, holds only the
round-off that the solve of the other components leaves.
A small step counts as no convergence: an iteration that stalls with a residual
above the test has not found the solution.

The line search, the divergence test and the printed residual use the scaled
norm

.. math::

   \lVert \mathbf{R} \rVert_s = \Bigl( \sum_v \bigl( \lVert \mathbf{R}_v \rVert
   / \bar{R}_v \bigr)^2 \Bigr)^{1/2} ,

with :math:`\bar{R}_v` the largest reference norm of equation :math:`v` in the solve.
A norm that added the raw residuals of several equations would follow the
equation with the largest numbers, which depends on the units.  The iteration
stops as diverged when an equation grows to more than :math:`10^6` times its
size at the first iterate, or after ``max_iterations`` steps (default 50).

Continuation
------------

When Newton's method does not converge on a steady problem, the solve starts
again from the starting state and ramps the data that drive the problem by a
fraction :math:`\ell` from 0 to 1.

* The sources, body forces and buoyancy, prescribed fluxes, tractions and
  pressures, point loads and the distributed loads of plates and beams are
  multiplied by :math:`\ell`.
* Each prescribed value moves from its starting value :math:`u_0` to
  :math:`u_0 + \ell \, (\hat{u} - u_0)`, with :math:`\hat{u}` the value the
  condition prescribes, so a wall heated from 300 K never passes through 0 K.

The converged solution of one step is the initial guess of the next.  The first
step is :math:`\ell = 1/2`.  A step that converges doubles the next increment,
and a step that fails halves it and starts again from the last converged state,
down to an increment of :math:`1/1024`.  A problem with no data to ramp, such as
one driven by a solution-dependent source alone, is not continued.  The steps of
a transient run are not continued: the time step controller retries a failed
step with a smaller time step instead.

Time integration
----------------

Split the residual into :math:`\mathbf{R}_t`, the terms of the time derivatives,
and :math:`\mathbf{R}_s`, the steady terms (the fluxes and the sources).  With
:math:`\Delta t` (s) the time step and :math:`\mathbf{U}^n` the state at the
time :math:`t^n`, a step of the :math:`\theta` method solves

.. math::

   \mathbf{R}_t(\mathbf{U}^{n+1})
   + \theta \, \mathbf{R}_s(\mathbf{U}^{n+1}, t^{n+1})
   + (1 - \theta) \, \mathbf{R}_s(\mathbf{U}^{n}, t^{n}) = 0 ,

with the dimensionless weight :math:`\theta`.  The one storage kernel,
``time_derivative``, gives :math:`\mathbf{R}_t` the term

.. math::
   :label: time-storage

   c \, \frac{m(u^{n+1}, \mathbf{q}^{n+1}) - m(u^n, \mathbf{q}^n)}{\Delta t} ,

with :math:`m` the storage, an expression of the unknown :math:`u`, the other
fields, the position, the time and the internal states :math:`\mathbf{q}` of a
``parsed_state`` object, and :math:`c` a coefficient.  The difference of the
storage itself makes the change of the stored amount over a step equal to what
the fluxes and the sources bring, for any storage :math:`m`, where the rate form
:math:`c\, \partial u/\partial t` of a capacity :math:`c` that varies with the
solution does not.  The default storage is :math:`m = u`, which gives the rate
form :math:`c\, (u^{n+1} - u^n)/\Delta t` with :math:`c` at the new state: the
heat capacity :math:`\rho c_p` of heat conduction, the density of the momentum
of a flow, or a capacity whose antiderivative has no closed form, such as one
given by a table.  For a constant :math:`c` the two forms are the same discrete
equation.

On a mesh of linear elements the automatic quadrature lumps the storage of an
equation of first order in time when no kernel of its variable carries it with
a flow: each node takes the integral of its shape function, which in DMCDM is
the measure of its control domain, with the storage at the node.  The lumped
storage keeps the maximum principle of diffusion for every step on a mesh
without obtuse angles [ThomeeWahlbin2008]_.  A quadratic element, whose lumped
weights are not all positive, a variable carried by a flow, whose transport
loses phase accuracy with a lumped storage, and the damping of a structure,
which is proportional to its consistent mass, take the Gauss rule.  The lumping
serves time stepping, so an eigenvalue study (:doc:`eigenvalues`) takes the
Gauss rule on every mesh.  Under FEM, each eigenvalue of a symmetric problem is
then an upper bound of the exact one (the Rayleigh-Ritz method).

The schemes are ``"backward_Euler"``
(:math:`\theta = 1`, the default, first order), ``"Crank-Nicolson"``
(:math:`\theta = 1/2`, the trapezoidal rule [CrankNicolson1947]_, second order)
and ``"forward_Euler"`` (:math:`\theta = 0`, explicit in the steady terms).

* The old part :math:`\mathbf{R}_s(\mathbf{U}^n, t^n)` is evaluated once per
  step, at the old time, so a source or a coefficient that depends on time
  explicitly takes its value at :math:`t^n` there.
* An equation without a time derivative, such as the mass equation of a flow or
  the equilibrium of a quasi-static solid, holds at the new time.  The code
  takes it, and the pressure gradient, at :math:`t^{n+1}` with the weight 1.
* A problem with no time derivative at all is marched quasi-statically: each
  step solves the steady problem at its time.
* ``"forward_Euler"`` refuses a stabilization whose residual holds the time
  derivative.
* A storage with ``quadrature="nodal"`` is lumped, and a storage with a Gauss
  rule is consistent on any mesh.  FEM refuses the lumped storage on ``Tri6``,
  ``Tet10``, ``Quad8`` and ``Hex20``, where the integral of a corner shape
  function is zero or negative.

Second-order equations
~~~~~~~~~~~~~~~~~~~~~~

An equation with a second time derivative, the inertia of a solid or the ``m``
of ``general_form_PDE``, takes the generalized-:math:`\alpha` method of
[ChungHulbert1993]_.  The code keeps the rate :math:`\dot{\mathbf{U}}` and the
acceleration :math:`\ddot{\mathbf{U}}` of the unknowns from step to step.  The
Newmark relations

.. math::

   \mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t \, \dot{\mathbf{U}}^n
     + \Delta t^2 \bigl( (\tfrac{1}{2} - \beta_\mathrm{N}) \ddot{\mathbf{U}}^n
     + \beta_\mathrm{N} \ddot{\mathbf{U}}^{n+1} \bigr) ,
   \qquad
   \dot{\mathbf{U}}^{n+1} = \dot{\mathbf{U}}^n
     + \Delta t \bigl( (1 - \gamma_\mathrm{N}) \ddot{\mathbf{U}}^n
     + \gamma_\mathrm{N} \ddot{\mathbf{U}}^{n+1} \bigr)

give :math:`\ddot{\mathbf{U}}^{n+1}` and :math:`\dot{\mathbf{U}}^{n+1}` as
functions of :math:`\mathbf{U}^{n+1}`, with the dimensionless Newmark parameters
:math:`\beta_\mathrm{N}` and :math:`\gamma_\mathrm{N}`.  The time-derivative
terms take the acceleration
:math:`(1 - \alpha_m) \ddot{\mathbf{U}}^{n+1} + \alpha_m \ddot{\mathbf{U}}^n`
and the rate
:math:`(1 - \alpha_f) \dot{\mathbf{U}}^{n+1} + \alpha_f \dot{\mathbf{U}}^n`, and
the steady terms take the weights of the :math:`\theta` method with
:math:`\theta = 1 - \alpha_f`.  The first-order equations of the same problem
take that :math:`\theta`.  A ``time_derivative`` of a second-order equation is
a damping :math:`c\, \dot{u}`, whose storage is the unknown itself, with the
rate above in place of :eq:`time-storage`.

The dimensionless parameters follow from the spectral radius
:math:`\rho_\infty` at infinite frequency (``spectral_radius``, default 0.75,
from 0 to 1):

.. math::

   \alpha_m = \frac{2\rho_\infty - 1}{\rho_\infty + 1} , \qquad
   \alpha_f = \frac{\rho_\infty}{\rho_\infty + 1} , \qquad
   \gamma_\mathrm{N} = \tfrac{1}{2} - \alpha_m + \alpha_f , \qquad
   \beta_\mathrm{N} = \tfrac{1}{4} (1 - \alpha_m + \alpha_f)^2 .

The method is second order and unconditionally stable, and it damps the modes
that the step does not resolve by the factor :math:`\rho_\infty` per step.
``"Newmark"`` is the average acceleration rule, :math:`\alpha_m = \alpha_f = 0`,
:math:`\beta_\mathrm{N} = 1/4` and :math:`\gamma_\mathrm{N} = 1/2`, which keeps
the energy of a linear undamped system.

At the start, the acceleration solves
:math:`\mathbf{M} \ddot{\mathbf{U}}^0 = -\mathbf{R}(\mathbf{U}^0,
\dot{\mathbf{U}}^0)`, with :math:`\mathbf{M}` the Jacobian of the second time
derivatives and :math:`\mathbf{R}` the residual without them.  A prescribed
value that changes in time sets its own rate and acceleration, by central
differences over :math:`10^{-3} \Delta t`.

Time step control
-----------------

The option ``time_stepper`` chooses how the step changes.

``"error"`` (the default)
   The step follows an estimate :math:`e` of the local truncation error.

``"fixed"``
   The step stays :math:`\Delta t` and lands on every output time and on the
   end time.  A target less than :math:`2\Delta t` away is reached in two
   equal steps, so no step is shorter than :math:`\Delta t / 2` unless two
   targets are closer than that.

``"iteration"``
   A step that converged in fewer than ``optimal_iterations`` minus
   ``iteration_window`` iterations (defaults 4 and 2) is followed by a step
   ``growth_factor`` times larger, and a step that needed more than
   ``optimal_iterations`` plus ``iteration_window`` by a step
   ``cutback_factor`` times smaller.

``"cfl"``
   Each step gives the Courant number ``courant_number`` at the velocity of the
   step before, grown by at most ``growth_factor``.  It needs the projection
   integration of a flow, which measures the Courant number (:doc:`fluids`).

With every stepper, a step whose nonlinear solve fails is discarded and retried
with the step times ``cutback_factor`` (default 0.5).  The steps land on the
output times.  The run stops with an error after ``max_rejected_steps``
rejections in a row (default 10) or below the smallest step, which is
:math:`10^{-6} \Delta t` by default.

The error estimate
~~~~~~~~~~~~~~~~~~

The estimate :math:`e` is dimensionless.  For two states :math:`\mathbf{U}_a`
and :math:`\mathbf{U}_b`, let :math:`\delta_v` be the largest difference of
variable :math:`v` at a node.  The difference of the two states is

.. math::

   \lVert \mathbf{U}_a - \mathbf{U}_b \rVert_e = \max_v \frac{\delta_v}{s_v} ,

over the variables with a time derivative, with :math:`s_v` the scale of
:math:`v`.  A variable without a time derivative, such as the pressure of an
incompressible flow, has no truncation error of its own and is left out.  A
problem without a time derivative measures every variable.

The scale is the largest of three sizes of :math:`v`:

* its range :math:`r_v`, the largest value less the smallest,
* its change :math:`c_v`, the largest difference at a node from the state at
  the start of the run, and
* the floor :math:`\sqrt{\epsilon_m}\, m_v`, with :math:`m_v` the largest
  magnitude of :math:`v` and :math:`\epsilon_m = 2^{-52}` the machine epsilon
  of double precision.

.. math::

   s_v = \max\bigl( r_v,\ c_v,\ \sqrt{\epsilon_m}\, m_v \bigr) .

Each size is the largest it has reached over the accepted states of the run and
the state :math:`\mathbf{U}_a` of the step.  The components of a vector
unknown, a velocity or a displacement, share one scale, the largest over the
components, as in the convergence test.

The change has no offset, so the scale of a temperature near
:math:`300\ \mathrm{K}` is the distance it has moved from its initial value.  A
field that is uniform, or zero up to round-off, has a range near zero and takes
the scale of its change: the displacement of a rigid motion, or the velocity of
a uniform flow.  The floor keeps the round-off of a field at rest from counting
as an error.  The sizes start again at each call of ``solve_transient``, so a
run that adapts the mesh starts them again on each new mesh.

For a step of the :math:`\theta` method, the code compares the step with the
polynomial through the last accepted states extrapolated to the end of the step
(the device of Milne).  Let :math:`\Delta t` be the step and
:math:`\Delta t_1` and :math:`\Delta t_2` the two steps before it.

* For an order-one member, :math:`\theta \ne 1/2`, the local error is
  :math:`(\theta - \tfrac12) \Delta t^2 \ddot{u}` and the linear predictor
  :math:`\mathbf{U}_p = \mathbf{U}^n + (\Delta t / \Delta t_1)(\mathbf{U}^n -
  \mathbf{U}^{n-1})` has the error
  :math:`-\ddot{u} \, \Delta t (\Delta t + \Delta t_1)/2`.  With
  :math:`c_\theta = \theta - \tfrac12`,

  .. math::

     e = \Bigl| \frac{2 c_\theta \Delta t}{2 c_\theta \Delta t + \Delta t
         + \Delta t_1} \Bigr| \, \lVert \mathbf{U}^{n+1} - \mathbf{U}_p \rVert_e .

* For Crank-Nicolson, the local error is :math:`\Delta t^3 \dddot{u} / 12` and
  the quadratic predictor through the last three states has the error
  :math:`-\dddot{u} \Pi / 6`, with
  :math:`\Pi = \Delta t (\Delta t + \Delta t_1)(\Delta t + \Delta t_1 + \Delta t_2)`,
  so

  .. math::

     e = \frac{\Delta t^3}{2\Pi + \Delta t^3} \,
         \lVert \mathbf{U}^{n+1} - \mathbf{U}_p \rVert_e .

For a step of the projection integration of a flow (:doc:`fluids`), let
:math:`q` be its order, the ``time_order`` :math:`\ell` up to 2.  The predictor
:math:`\mathbf{U}_p` is the polynomial of degree :math:`q` through the last
:math:`q + 1` accepted states, extrapolated to the end of the step, and the
corrector is BDF\ :math:`q`.  Let :math:`t^{n+1}` be the end of the step,
:math:`t^{n}, \dots, t^{n-q}` the times of the accepted states, and
:math:`b_0` the coefficient of the new state in BDF\ :math:`q` on the levels
:math:`t^{n+1}, \dots, t^{n-q+1}`.  With :math:`A` the derivative of order
:math:`q + 1` of the velocity over :math:`(q + 1)!`, the corrector has the local
error :math:`(\Delta t / b_0) A \prod_{m=0}^{q-1} (t^{n+1} - t^{n-m})` and the
predictor the error :math:`-A \prod_{m=0}^{q} (t^{n+1} - t^{n-m})`, so

.. math::

   e = \frac{\Delta t}{\Delta t + b_0 \, (t^{n+1} - t^{n-q})} \,
       \lVert \mathbf{U}^{n+1} - \mathbf{U}_p \rVert_e .

For BDF1 this is the estimate of the order-one :math:`\theta` method with
:math:`\theta = 1`.  ``time_order`` 3 takes the pair of order 2, the order at
which its velocity converges because of the splitting.  The estimate measures
the error of the backward difference.  The splitting error of the pressure is
outside it.

The first steps, before enough states are accepted, and an equation with a
second time derivative take step doubling instead [RichardsonGaunt1927]_.  The
code advances the interval once with one step and once with two half steps,
keeps the solution of the half steps, and estimates

.. math::

   e = \frac{\lVert \mathbf{U}_{1/2} - \mathbf{U}_1 \rVert_e}{2^p - 1} ,

with :math:`\mathbf{U}_1` the state after the single step,
:math:`\mathbf{U}_{1/2}` that after the two half steps and :math:`p` the order
of the scheme: 1 for an order-one :math:`\theta` method, 2 for Crank-Nicolson
and for the generalized-:math:`\alpha` and Newmark methods, and :math:`q` for
the projection integration.

The controller [HairerNorsettWanner1993]_ accepts the step when :math:`e` is at
most :math:`\epsilon_t`, the ``error_tolerance`` (default :math:`10^{-3}`).  With
the factor :math:`f_q = 0.9 \, (\epsilon_t / e)^{1/(p+1)}` and the growth factor
:math:`f_g` (``growth_factor``, default 2), the next step is

.. math::

   \Delta t_\mathrm{new} =
   \begin{cases}
     \Delta t \, \min\bigl(f_g, \max(1, f_q)\bigr) & \text{after an accepted step,} \\
     \Delta t \, \max(0.1, f_q) & \text{after a step rejected by its error.}
   \end{cases}

The factor :math:`f_q` is the step at which the estimate of order :math:`p + 1` in
:math:`\Delta t` meets :math:`\epsilon_t`, with the safety factor 0.9, so one retry
usually suffices.  A step whose nonlinear solve fails has no estimate and takes
``cutback_factor``.

Linear solvers
--------------

PETSc solves each linear system
:math:`\mathbf{J} \, \delta\mathbf{U} = -\mathbf{R}`.  Without
``petsc_options``, the code chooses the method from the structure of the
system, and :ref:`the choice of linear solver <linear-solver-choice>` lists the
choices.  This section gives the theory behind them.

The unknowns are numbered node by node: the unknown of variable :math:`v` at
node :math:`a` is the entry :math:`a \, n_v + v`, with :math:`n_v` the number of
variables, and :math:`n_v` is the block size of the matrix.  This keeps the
couplings of a multi-field model close to the diagonal.  The row of a prescribed
value takes the mean size of the diagonal of its equation, so a change of units
scales it with the rest of that equation.

Within the nonlinear iteration, PETSc solves the scaled system
:math:`\mathbf{D} \mathbf{J} \, \delta\mathbf{U} = -\mathbf{D} \mathbf{R}`.  The
diagonal matrix :math:`\mathbf{D}` holds :math:`1 / s_e` on every row of equation
:math:`e`, the rows of its prescribed values included.  For each variable
:math:`v`, the code takes the larger of
:math:`\lVert \mathbf{R}^{\mathrm{ref}}_v \rVert` at the current iterate and
:math:`\bar{R}_v`.  The size :math:`s_e` is the Euclidean norm of these values over
the variables of the equation, with the components of a vector unknown taken
together as in the convergence test.  An equation whose terms have all been zero
so far keeps its rows as they are.  The Krylov method stops when the norm of the scaled residual of the linear
system falls to the linear tolerance times
:math:`\lVert \mathbf{D} \mathbf{R} \rVert`.  Each equation then counts by its
own size, and the test has no unit.  A raw norm adds equations in different units
and follows the one with the largest numbers.  The scaling leaves
:math:`\delta\mathbf{U}` as it is, and the block preconditioners of a flow build
their operators for the scaled matrix.  The size :math:`s_e` changes only while
the terms of the equation grow, so the tolerance of Eisenstat and Walker
[EisenstatWalker1996]_ compares the residuals of successive iterates in one norm
once the sizes settle.

The limits of the direct solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A direct factorization of a sparse matrix from a mesh fills in.  With a good
ordering, the factors of a two-dimensional problem with :math:`n` unknowns hold
:math:`O(n \log n)` entries and cost :math:`O(n^{3/2})` operations, and those of
a three-dimensional problem hold :math:`O(n^{4/3})` entries and cost
:math:`O(n^2)` operations [George1973]_ [LiptonRoseTarjan1979]_.  A
preconditioned Krylov iteration costs a few products with the matrix, of
:math:`O(n)` operations each, per iteration.  The code therefore factorizes
every system in one dimension, systems up to :math:`10^5` unknowns in two
dimensions and up to 4000 unknowns in three, and iterates on larger ones.  An
iteration that does not converge in 500 iterations hands its iterate to the
direct solver.

The saddle-point system of a flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Jacobian of an incompressible flow with the pressure :math:`p` as an
unknown has the block form

.. math::

   \begin{bmatrix} \mathbf{A}_v & \mathbf{G} \\ \mathbf{D} & \mathbf{C}
   \end{bmatrix}
   \begin{bmatrix} \delta\mathbf{U}_v \\ \delta\mathbf{U}_p \end{bmatrix}
   = - \begin{bmatrix} \mathbf{R}_v \\ \mathbf{R}_p \end{bmatrix} ,

with :math:`\mathbf{A}_v` the momentum block, :math:`\mathbf{G}` the pressure
gradient, :math:`\mathbf{D}` the discrete divergence and :math:`\mathbf{C}` the
pressure block.  The subscript :math:`v` marks the velocities, with any other
unknown such as a temperature, and :math:`p` the pressure.  The pressure block
is zero for the Taylor-Hood element and the pressure Laplacian of the
stabilization for an equal-order element.  The preconditioner is the block upper
triangle

.. math::

   \mathbf{P} = \begin{bmatrix} \mathbf{A}_v & \mathbf{G} \\ \mathbf{0} &
   \hat{\mathbf{S}} \end{bmatrix} .

With it, the number of iterations of the generalized minimal residual method
(GMRES) depends on how well
:math:`\hat{\mathbf{S}}` approximates the Schur complement
:math:`\mathbf{S} = \mathbf{C} - \mathbf{D} \mathbf{A}_v^{-1} \mathbf{G}`,
which is dense.  Up to its sign, which the code takes from the coupling blocks of
the matrix, :math:`\mathbf{S} \approx \mathbf{G}^T \mathbf{A}_v^{-1}
\mathbf{G}`.  The option ``-dualmesh_schur`` in ``petsc_options`` chooses one
of three approximations, and a large flow takes ``lsc`` by default.

The approximations use these operators on the pressure space, with
:math:`N_a` the pressure shape functions, :math:`\rho` the density, :math:`\mu`
the dynamic viscosity, :math:`\Delta t` the time step and :math:`\theta` the
weight of the time scheme (:math:`1/\Delta t = 0` and :math:`\theta = 1` in a
steady solve):

* :math:`\mathbf{M}_p`, the pressure mass matrix, with the entries
  :math:`\int N_a N_b \, \mathrm{d}V`,
* :math:`\mathbf{K}_p`, the pressure diffusion matrix, with the entries
  :math:`\int \nabla N_a \cdot \nabla N_b \, \mathrm{d}V`,
* :math:`\mathbf{N}_p(\mathbf{v})`, the pressure advection matrix at the current
  velocity :math:`\mathbf{v}`, with the entries
  :math:`\int N_a \, \mathbf{v} \cdot \nabla N_b \, \mathrm{d}V`,
* :math:`\mathbf{Q}_v` and :math:`\mathbf{Q}_p`, the lumped velocity and
  pressure mass matrices, each the diagonal of the mass matrix of an element
  scaled to the measure of the element,
* :math:`\mathbf{A}_p = \mathbf{G}^T \mathbf{Q}_v^{-1} \mathbf{G}`, plus the
  stabilization :math:`\mathbf{C}_1` below for an equal-order element.

``pressure_mass``
   :math:`\hat{\mathbf{S}} = \tfrac12 (\mathbf{C} + \mathbf{C}^T) \pm
   \mathbf{M}_{p,\mu}`, the symmetric part of the pressure block plus, with the
   sign that the code takes from the coupling blocks, the pressure mass matrix
   scaled by the inverse viscosity, with the entries :math:`\int N_a N_b / \mu \, \mathrm{d}V`.  For
   the Stokes equations :math:`\mathbf{M}_{p,\mu}` is spectrally equivalent to
   :math:`\mathbf{S}`, with bounds independent of the element size.  The
   approximation leaves out convection and the time derivative.

``pcd``
   The pressure convection-diffusion preconditioner [KayLoghinWathen2002]_ in
   the form of [ElmanTuminaro2009]_,

   .. math::

      \hat{\mathbf{S}}^{-1} = \mathbf{A}_p^{-1} \mathbf{H}_p \mathbf{Q}_p^{-1} ,
      \qquad
      \mathbf{H}_p = \frac{\rho}{\Delta t} \mathbf{M}_p
        + \theta \bigl( \mu \mathbf{K}_p + \rho \, \mathbf{N}_p(\mathbf{v})
        \bigr) ,

   with :math:`\mathbf{H}_p` the operator of the momentum equation on the
   pressure space.  On the sides where every velocity component is prescribed,
   :math:`\mathbf{H}_p` carries the Robin condition
   :math:`\mu \, \partial p / \partial n = \rho \, (\mathbf{v} \cdot \mathbf{n})
   \, p`, which adds :math:`\theta \rho \max(0, -\mathbf{v} \cdot \mathbf{n})
   \int N_a N_b \, \mathrm{d}S`.  It is zero on a wall and acts at an inflow.

``lsc``
   The least squares commutator
   [ElmanHowleShadidShuttleworthTuminaro2006]_,

   .. math::

      \hat{\mathbf{S}}^{-1} = \mathbf{A}_p^{-1} \bigl( \mathbf{G}^T
      \mathbf{Q}_v^{-1} \mathbf{A}_v \mathbf{Q}_v^{-1} \mathbf{G} +
      \mathbf{C}_2 \bigr) \mathbf{A}_p^{-1} ,

   which needs no operator besides the blocks of the matrix, with the
   stabilization :math:`\mathbf{C}_2` for an equal-order element.

For an equal-order element, :math:`\mathbf{G}^T \mathbf{Q}_v^{-1} \mathbf{G}`
holds the spurious pressure modes of the element in its null space.  The
stabilization of [ElmanHowleShadidSilvesterTuminaro2007]_ removes them with the
:math:`L^2` distance of the pressure from its mean on each element :math:`k`,

.. math::

   \mathbf{C}_1 = c_1 \sum_k h_k^{-2} \, \mathbf{E}_k , \qquad
   \mathbf{C}_2 = \sum_k \Bigl( \frac{\theta \mu}{h_k^4}
     + \frac{\rho}{\Delta t \, h_k^2} \Bigr) \mathbf{E}_k , \qquad
   \mathbf{E}_k = \mathbf{M}_k - \frac{\mathbf{m}_k \mathbf{m}_k^T}{|k|} ,

with :math:`\mathbf{M}_k` the pressure mass matrix of the element,
:math:`\mathbf{m}_k` the integrals of its pressure shape functions, :math:`|k|`
its measure and :math:`h_k = |k|^{1/n_\mathrm{d}}` its size in the dimension
:math:`n_\mathrm{d}` of the mesh.  The factor :math:`c_1` is 1 for ``lsc`` and
100 for ``pcd``, which applies :math:`\mathbf{A}_p^{-1}` once, to a vector that
keeps the spurious modes at full size.  :math:`\mathbf{A}_p` is taken from the
matrix, because an assembled pressure Laplacian would need a Robin condition at
an outflow.

PETSc's Schur complement field split applies the preconditioner, with the
velocities and any other unknown in the first split and the pressure in the
second.  The defaults are flexible GMRES, which accepts a preconditioner that
changes between iterations [Saad1993]_, the upper block-triangular
factorization, and one V-cycle of the algebraic multigrid BoomerAMG
[FalgoutYang2002]_ for :math:`\mathbf{A}_v` and for :math:`\mathbf{A}_p`, or
for :math:`\hat{\mathbf{S}}` with ``pressure_mass``, with
the coarsening and the interpolation of [DeSterckYangHeys2006]_ and
[DeSterckFalgoutNoltingYang2008]_.  A prescribed pressure is a row of the
diagonal, and its Schur complement is its own entry.

The split by physics
~~~~~~~~~~~~~~~~~~~~

The option ``-dualmesh_split physics``, the default for a solid coupled to other
physics, makes each physics a block: the displacements of a solid together, and
every other variable alone.  The block lower-triangular preconditioner solves the
blocks one after the other, in the order in which the physics were added, each
with one cycle of smoothed aggregation multigrid on its diagonal block, and moves
the coupling to the earlier blocks to the right-hand side.  A coupling that one
physics drives, such as the thermal expansion of a solid by a temperature, is
then exact in the preconditioner when that physics comes first.  The split
refuses a flow, whose velocity and pressure need ``-dualmesh_schur``.

Smoothed aggregation with the rigid body modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solid alone takes GMRES with PETSc's smoothed aggregation multigrid (GAMG)
[VanekMandelBrezina1996]_.  It groups the nodes into aggregates of strongly
connected neighbours, and on each aggregate the coarse basis represents the
near null space of the operator exactly: the vectors on which the operator
nearly vanishes.  The code gives PETSc the three translations and three
rotations of a rigid body for each solid, two translations and one rotation in
two dimensions, with the rotations in Cartesian coordinates only, and the
constant for every other variable.  GMRES with GAMG or BoomerAMG is
preconditioned from the right, so it measures the true residual of the system.

:doc:`eigenvalues` gives the eigenvalue study, which takes the same Jacobians.

:doc:`/verification` gives the cases that check the iterations, the time
schemes, the step control and the linear solvers.
