Solving
=======

This page shows how to solve a problem once you have defined it: a steady
solve, a transient solve, the settings of the nonlinear and the linear
solvers, the remedies when a solve fails, and runs on several threads or
processes.  :doc:`/theory/nonlinear_and_time` gives the theory of the
nonlinear methods and of the time schemes, and :doc:`/theory/parallel` gives
the theory of the parallel runs and of the linear solvers.

The Portable, Extensible Toolkit for Scientific Computation (PETSc)
[PETSc2023]_ solves the nonlinear and the linear systems, in the same way on
one process and on several.

.. contents::
   :local:
   :depth: 2

Where the settings go
---------------------

You give the settings of a solve in one of two places:

- In the input, which ``dualmesh.problem(...)`` builds in Python and a YAML or
  TOML case file holds for ``dualmesh run``.  The ``solver`` section sets the
  tolerance of the nonlinear solve and the PETSc options, and the ``time``
  section makes the analysis transient.  :doc:`input` lists every key.
- At the object level, :meth:`~dualmesh.Problem.solve` and
  :meth:`~dualmesh.Problem.solve_transient` take every setting on this page as
  a keyword argument.  An unknown keyword raises a ``TypeError`` that lists
  the known options of the nonlinear and the linear solve.

The other settings on this page, such as ``nonlinear_solver`` or
``max_iterations``, exist at the object level only.

Solve a steady problem
----------------------

From the input
~~~~~~~~~~~~~~

Without a ``time`` section, ``solve`` of the input solves the steady problem.
This square plate conducts heat between two sides held at different
temperatures, and its ``solver`` section asks for a tighter tolerance and a
direct linear solver:

.. code-block:: python

   import dualmesh

   plate = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8]}})
   plate.physics("heat_conduction")
   plate.properties(thermal_conductivity="45 W/(m*K)")
   plate.boundary_conditions("left", temperature="400 K")
   plate.boundary_conditions("bottom", temperature="300 K")
   plate.solver(relative_tolerance=1.0e-8, PETSc_options={"ksp_type": "preonly", "pc_type": "lu"})
   solution = plate.solve()

In a case file, the same ``solver`` section reads:

.. code-block:: yaml

   solver:
     relative_tolerance: 1.0e-8
     PETSc_options: {ksp_type: preonly, pc_type: lu}

``relative_tolerance`` is the tolerance of `Set the convergence test`_.
``PETSc_options`` maps PETSc option names, written without the leading dash,
to their values as text or numbers, and takes every option of
`Choose the linear solver`_.  ``true`` gives a flag without a value, as the
``petsc_options`` of Python do.  The solve refuses ``snes_rtol``, ``snes_atol``
and ``snes_stol``, which the convergence test does not read.

The ``solve`` of the input prints one line that names the solve and then the
result (``report="summary"``, the default).  ``report="full"`` also prints the
problem and every iteration, and ``report="none"`` prints nothing.

From Python
~~~~~~~~~~~

.. code-block:: python

   result = problem.solve()
   print(result.converged, result.total_iterations)

:meth:`~dualmesh.Problem.solve` assembles the residual :math:`\mathbf{R}` of
the vector :math:`\mathbf{U}` of the nodal unknowns and its Jacobian
:math:`\mathbf{J} = \partial \mathbf{R} / \partial \mathbf{U}`, solves the
linear system :math:`\mathbf{J}\, \delta\mathbf{U} = -\mathbf{R}` for the
update :math:`\delta\mathbf{U}`, and repeats until the convergence test
passes.  It leaves the solution on the problem, where
:meth:`~dualmesh.Problem.values` reads it, and returns a
:class:`~dualmesh.SolveResult` (see `Read the result`_).

``report`` sets what the solve prints: ``"full"`` (the default: the problem
with every parameter, the defaults marked, the iterations and the result),
``"summary"`` or ``"none"``.  ``output``, an :class:`~dualmesh.Output` group,
writes the fields once, at the end (see :doc:`output`).

A steady flow that starts at rest takes its first Newton iterate from the
Stokes solution, the solution of the same problem without the inertia.

Choose the nonlinear method
---------------------------

``nonlinear_solver`` takes one of three values, written in upper or lower
case:

``"Newton"``
   The default.  Each iteration solves with the exact Jacobian, which every
   kernel provides by forward-mode automatic differentiation [Wengert1964]_.
   Near the solution, the number of correct digits doubles at each iteration.

``"Picard"``
   Direct iteration.  Each iteration evaluates the nonlinear coefficients at
   the last iterate and solves the linear problem that results.  It converges
   linearly, at a lower cost per iteration than Newton's method, and it often
   converges from a start where Newton's method diverges.

``"linear"``
   One assembly and one linear solve, followed by the convergence test.  Use
   it for a linear problem.  When the residual after the solve fails the test,
   the solve stops with an error: either the problem is nonlinear or the
   linear solver did not converge.

``line_search`` applies to Newton's method:

``"automatic"``
   The default.  Backtracking when a kernel needs it, which is the case of the
   ``limited`` advection scheme, whose limiter has kinks on which the full
   Newton step can cycle.  No line search otherwise.

``"backtracking"``
   Halves the Newton step until the scaled residual norm falls, at most
   ``max_line_search_steps`` times (default 10).  Each trial costs one
   assembly of the residual.

``"none"``
   Takes the full Newton step.

``relaxation`` applies to an iteration without a line search, Picard or
Newton.  It is a weight :math:`\gamma` from 0 to below 1, with the default 0
for no relaxation.  From the second iteration on, the new iterate is

.. math::

   \mathbf{U}^{(r+1)} = (1 - \gamma)\, \tilde{\mathbf{U}}^{(r+1)} + \gamma\, \mathbf{U}^{(r)} ,

with :math:`r` the iteration number and :math:`\tilde{\mathbf{U}}^{(r+1)}`
the iterate that the full step reaches, so that each update takes the
fraction :math:`1 - \gamma` of its step.  The code refuses a value outside
this range.  Relaxation damps an iteration that oscillates without settling.

Set the convergence test
------------------------

The iteration converges when the residual of every equation is at most
``relative_tolerance`` (default :math:`10^{-10}`) times the size of the terms
of that equation, or at the level of round-off of its rows.  An equation is
the set of rows of one variable.  The test has no unit, so a problem
converges in the same iterations whatever units it is written in.

``max_iterations`` (default 50) limits the number of iterations.
``error_on_divergence`` (default true) sets what happens when the solve does
not converge: with true, it raises an exception, and with false, it returns
the result with ``converged`` false.  Set it to false in a loop of your own
that expects some solves to fail.

The iteration also stops when the residual grows to more than :math:`10^6`
times its first value, or when the residual or the Jacobian has an entry that
is not finite.

Continuation
------------

When Newton's method does not converge on the problem itself, a steady solve
applies the data in increments, each solve starting from the solution of the
last one.  It ramps the data that drive the problem (the sources, the body
forces and buoyancy, the prescribed fluxes, tractions and loads) up from
zero, and the prescribed values from those of the starting state.  The terms
of stiffness, conduction and viscosity keep their full value.

The first increment is half the data.  An increment that converges doubles
the next one, and an increment that fails is halved and taken again from the
last converged state.  The continuation stops at the full data, or when the
increment falls below :math:`1/1024` of the data.  It needs no setting, and
the solve with ``nonlinear_solver="linear"`` does not take it.  The error
message of a failed continuation states the fraction of the data it reached,
and ``result.history`` records each increment with its ``continuation_step``
and ``data_fraction``.

.. _linear-solver-choice:

Choose the linear solver
------------------------

Each iteration solves a linear system.  The solver chooses the method itself,
or takes the one that your PETSc options give.

The automatic choice
~~~~~~~~~~~~~~~~~~~~

Without PETSc options of the linear solver, the solver chooses a method from
the structure of the system.  The choices are iterations of the generalised
minimal residual method (GMRES) [Saad1993]_, or of flexible GMRES, which
allows a preconditioner that changes from one iteration to the next.  Their
preconditioners are PETSc's geometric algebraic multigrid (GAMG), the
algebraic multigrid BoomerAMG of the hypre library [FalgoutYang2002]_, and
the splits of `Split a flow or a coupled problem`_:

.. list-table::
   :header-rows: 1

   * - System
     - Solver
   * - One dimension, two dimensions up to :math:`10^5` unknowns, three
       dimensions up to 4000 unknowns
     - The direct factorisation
   * - A pressure-velocity flow
     - Flexible GMRES with the Schur complement split ``lsc``
   * - Variables of different element orders, other than a flow
     - The direct factorisation
   * - A solid alone
     - GMRES with GAMG and the rigid body modes
   * - A solid coupled to other physics
     - Flexible GMRES with the split by physics
   * - Any other system
     - GMRES with BoomerAMG

The counts of unknowns are those of the whole problem, over all processes.

The direct factorisation is the MUltifrontal Massively Parallel sparse direct
Solver (MUMPS) [Amestoy2001]_ when PETSc has it, on one process and on
several.  MUMPS pivots, so it factorises the saddle point systems of the
Taylor-Hood element, which have zeros on the diagonal.  Without MUMPS, the solver
takes PETSc's own lower-upper (LU) factorisation on one process, which does
not pivot, and block Jacobi with incomplete LU on several processes.  A
direct solve that you choose with ``-pc_type lu`` also takes MUMPS, unless
``-pc_factor_mat_solver_type`` names another package.

When an iterative choice does not converge in 500 iterations, the solve goes
on with the direct factorisation from the last nonlinear iterate.  GMRES with
GAMG or BoomerAMG is preconditioned from the right, so it measures the true
residual of the system.  BoomerAMG takes the HMIS coarsening and the
extended+i interpolation of the splits of `Split a flow or a coupled problem`_,
and GAMG takes its place when PETSc has no hypre.  The GMRES methods that the code chooses repeat the
Gram-Schmidt orthogonalization of a new basis vector wherever one pass loses
orthogonality, which keeps a solve to :math:`10^{-12}` from stalling on rounding
errors.

Within a nonlinear iteration, the automatic iterative solvers take the tolerance of
Eisenstat and Walker [EisenstatWalker1996]_, which is loose while the
residual is large and tightens as it falls.  A direct solve, a solver you
choose with ``petsc_options`` and the solve of ``nonlinear_solver="linear"``
take ``linear_tolerance`` instead (default :math:`10^{-12}`, relative to the
norm of the right-hand side).  Both norms take each equation over the size of its
terms, so the test has no unit (:doc:`/theory/nonlinear_and_time`).

Give PETSc options
~~~~~~~~~~~~~~~~~~

``petsc_options`` takes PETSc's options as the string PETSc parses or as a
dictionary.  In a dictionary, the value ``None`` or ``True`` gives an option
without a value, and the value ``False`` leaves the option out:

.. code-block:: python

   problem.solve(petsc_options="-ksp_type preonly -pc_type lu")
   problem.solve(petsc_options={"ksp_type": "cg", "pc_type": "gamg"})

The options that start with ``-snes_`` go to SNES (Scalable Nonlinear
Equations Solvers), the nonlinear solver of PETSc that runs the iteration,
for example ``-snes_monitor`` or ``-snes_ksp_ew`` for the tolerance of
Eisenstat and Walker with a solver of your choice.  The convergence test stays
that of ``relative_tolerance``.  The other options replace the automatic
choice of the linear solver, and a solver you choose this way must converge
within 5000 iterations.  The options of one solve live in a private options
database, so they apply to that solve alone.

The unknowns are numbered node by node, all variables of a node together, and
the number of variables is the block size of the matrix, which PETSc's field
split preconditioners read.  For a two-dimensional flow, whose variables are
``velocity_x``, ``velocity_y`` and ``pressure`` in that order, PETSc's own
Schur complement split with algebraic multigrid on both blocks is:

.. code-block:: python

   options = (
       "-ksp_type fgmres"
       " -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2"
       " -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper"
       " -pc_fieldsplit_schur_precondition selfp"
       " -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type hypre"
       " -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre"
   )
   problem.solve(petsc_options=options)

Split a flow or a coupled problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two options of dualmesh, which PETSc reads among its own, choose
a split that PETSc cannot name.  A split you choose must converge within 5000
iterations.

``-dualmesh_schur lsc``, ``pcd`` or ``pressure_mass``
   The Schur complement split of a pressure-velocity flow.  The velocities,
   with any other unknown such as a temperature, form the split ``0``, and the
   pressure forms the split ``1``.  The preconditioner is the upper block
   triangle of the Jacobian, with an approximation of the Schur complement of
   the pressure:

   - ``lsc``, the least squares commutator
     [ElmanHowleShadidShuttleworthTuminaro2006]_, built from the blocks of the
     matrix alone.  The automatic choice takes it for a large flow.
   - ``pcd``, the pressure convection-diffusion preconditioner
     [KayLoghinWathen2002]_ in the form of [ElmanTuminaro2009]_, built from
     the operator of the momentum equation on the pressure space.
   - ``pressure_mass``, the pressure mass matrix divided by the dynamic
     viscosity, with the symmetric part of the pressure block.  It ignores
     convection, so it suits the Stokes equations.

   The equal-order element takes the stabilisation of
   [ElmanHowleShadidSilvesterTuminaro2007]_ in ``lsc`` and ``pcd``.  The
   defaults are flexible GMRES outside, the upper block-triangular
   factorisation, and one V-cycle of BoomerAMG for the momentum block and for
   the pressure operator: the pressure Laplacian of ``lsc`` and ``pcd``
   (options prefix ``fieldsplit_1_laplacian_``) or the approximation of
   ``pressure_mass`` (options prefix ``fieldsplit_1_``).  BoomerAMG takes the
   HMIS coarsening [DeSterckYangHeys2006]_ and the extended+i interpolation
   [DeSterckFalgoutNoltingYang2008]_, and GAMG takes its place when PETSc has
   no hypre.  Your options refine these defaults,
   for example ``{"dualmesh_schur": "lsc", "fieldsplit_0_pc_type": "lu"}`` for
   an exact momentum solve.  With ``lsc`` and ``pcd``, the approximation is
   itself the preconditioner of the split ``1``, so the options
   ``pc_fieldsplit_schur_precondition`` and ``fieldsplit_1_pc_type`` have no
   effect.

``-dualmesh_split physics``
   The split by physics.  The displacements of a solid form one block, and
   every other variable forms a block of its own.  The preconditioner solves
   the blocks one after the other, in the order in which you added the
   physics, each with one cycle of GAMG on its diagonal block.  A coupling
   that one physics drives, such as the thermal expansion of a solid by a
   temperature, is exact in the preconditioner when you add that physics
   first.  A flow is refused, because its velocity and pressure need
   ``-dualmesh_schur``, which takes any other unknown into its first block.

The two options exclude each other.

Use the rigid body modes
~~~~~~~~~~~~~~~~~~~~~~~~

GAMG builds its coarse problems by smoothed aggregation
[VanekMandelBrezina1996]_.  On each aggregate of nodes, the coarse basis
represents the near null space of the operator exactly: the constant for a
diffusion equation, and the rigid translations and rotations of a body for
elasticity (three and three in three dimensions, two and one in two).
dualmesh builds the near null space from the displacement
variables of the solids and the constant of every other variable.  It gives
the near null space to PETSc whenever the problem has the displacements of a
solid, also when your options choose GAMG.

Hand the system to another solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.Problem.linear_system` returns the residual and the Jacobian
of the current Newton step as a NumPy array and a SciPy sparse matrix, for a
solver outside PETSc.

Read the result
---------------

A :class:`~dualmesh.SolveResult` holds:

``converged``
   Whether the solve met the convergence test.

``total_iterations`` and ``linear_iterations``
   The nonlinear iterations and the iterations of the linear solver, summed
   over the continuation increments and the time steps.

``history``
   One record per nonlinear iteration, with ``continuation_step``,
   ``data_fraction``, ``iteration``, ``residual_norm`` and ``step_norm``.
   Look at it first when a solve fails.

``time_steps``, ``rejected_steps``, ``step_history`` and ``next_time_step``
   Filled by a transient solve (see `Control the time step`_).

Solve a transient problem
-------------------------

From the input
~~~~~~~~~~~~~~

A ``time`` section makes the analysis transient.  This steel slab, at
:math:`573\ \mathrm{K}`, has both faces held at :math:`293\ \mathrm{K}` from
the time zero (``examples/transient_slab.py``):

.. code-block:: python

   import dualmesh

   quench = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 40}})
   quench.physics("heat_conduction")
   quench.properties(
       thermal_conductivity="45 W/(m*K)",
       density="7850 kg/m^3",
       specific_heat_capacity="490 J/(kg*K)",
   )
   quench.initial_conditions(temperature="573 K")
   quench.boundary_conditions("left", temperature="293 K")
   quench.boundary_conditions("right", temperature="293 K")
   quench.time(end_time="40 s", initial_time_step="0.1 s", step_control={"by": "fixed"})
   solution = quench.solve()

The ``time`` section takes three keys:

``end_time`` (required)
   The time at which the run ends, a positive quantity in seconds.

``initial_time_step`` (required)
   The first time step, a positive quantity in seconds.

``step_control``
   How the step changes, by ``by`` and the optional key that goes with it:

   - ``{by: error, relative_tolerance: 1.0e-3}``, the default: the step keeps
     the estimated error of a step at the tolerance, a number above 0 and at
     most 1.  The default tolerance is :math:`10^{-3}`.
   - ``{by: fixed}``: the step stays at ``initial_time_step``, apart from the
     retries of a step that fails to converge.
   - ``{by: iterations, target_iterations: 4}``: the step follows the number
     of nonlinear iterations of a step.  The default target is 4.

   The defaults are those of ``solve_transient``, which the input leaves to
   it.  `Control the time step`_ describes each of them.

The input takes the default time scheme of the problem (see
`Choose the time scheme`_).

From Python
~~~~~~~~~~~

.. code-block:: python

   result = problem.solve_transient(end_time=100.0, time_step=1.0, time_scheme="Crank-Nicolson")

:meth:`~dualmesh.Problem.solve_transient` marches the solution from
``start_time`` (default 0) to ``end_time``, with ``time_step`` the first
step, which the ``"fixed"`` stepper keeps.  It also takes every
setting of the steady solve, which applies to the nonlinear solve of each
step.  A step does not take continuation.

``output``, an :class:`~dualmesh.Output` group, writes the fields at its
output times, on which the steps land (see :doc:`output`).
:meth:`~dualmesh.Problem.set_time_step_callback` sets a function
``callback(time, problem)`` that runs after every accepted step.
``max_steps`` (default 0, no limit) stops the run after that many accepted
steps, with ``result.next_time_step`` the step that the step control chose
next.  ``restart`` continues a run from a checkpoint file of an
:class:`~dualmesh.Output` group with ``checkpoint_interval``.  A problem with
inertia does not restart from a checkpoint.

A problem without a time derivative is marched quasi-statically: each step
solves the steady problem at its time.  This follows loads that change in
time and the history of a creeping material.

Choose the time scheme
~~~~~~~~~~~~~~~~~~~~~~

For a problem with first time derivatives, ``time_scheme`` selects a member
of the :math:`\theta` method, which weights the steady part
:math:`\mathbf{R}_{\mathrm{steady}}` of the residual between the old state
:math:`\mathbf{U}^{n}` at the time :math:`t^{n}` and the new state
:math:`\mathbf{U}^{n+1}` at :math:`t^{n+1} = t^{n} + \Delta t`:

.. math::

   \mathbf{R}_{\mathrm{time}}(\mathbf{U}^{n+1})
   + \theta\, \mathbf{R}_{\mathrm{steady}}(\mathbf{U}^{n+1}, t^{n+1})
   + (1 - \theta)\, \mathbf{R}_{\mathrm{steady}}(\mathbf{U}^{n}, t^{n}) = 0 ,

where :math:`\mathbf{R}_{\mathrm{time}}` holds the time derivatives and
:math:`\theta` is the weight of the new state.  The name of the scheme may be
written in lower case.

``"backward_Euler"``, :math:`\theta = 1`
   The default.  First-order accurate, stable for every step, and strongly
   damping.

``"Crank-Nicolson"``, :math:`\theta = 1/2`
   [CrankNicolson1947]_.  Second-order accurate and stable for every step, but
   without damping, so a sharp initial transient can leave oscillations that
   persist for several steps.

``"forward_Euler"``, :math:`\theta = 0`
   Explicit in the steady terms, and stable only below a critical step.  It
   refuses a stabilisation whose residual holds the time derivative.

An equation without a time derivative, such as the mass equation of an
incompressible flow or the equilibrium of a quasi-static solid, is a
constraint of the new time.  Every :math:`\theta` takes it at
:math:`t^{n+1}` with the weight 1, and so takes the pressure gradient of a
flow.

A problem with inertia, such as a solid with a ``density``, has second time
derivatives and takes one of two schemes:

``"generalized_alpha"``
   The default for such a problem: the method of Chung and Hulbert
   [ChungHulbert1993]_, second-order accurate.  ``spectral_radius``, the
   spectral radius :math:`\rho_\infty` at infinite frequency, sets the damping
   of the modes the mesh cannot resolve, from 0 (the highest frequency damped
   out in one step) to 1 (no damping), with the default 0.75.

``"Newmark"``
   The average acceleration rule: second-order accurate, without numerical
   damping, and conserving the energy.

The first-order equations of such a problem take the :math:`\theta` method
with :math:`\theta = 1 - \alpha_f`, where
:math:`\alpha_f = \rho_\infty / (\rho_\infty + 1)` for the
generalized-:math:`\alpha` method and :math:`\alpha_f = 0` for Newmark.  This
bar, fixed at one end, starts with a velocity that excites its first axial
mode (from ``tests/python/test_dynamics.py``):

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(0, 1.0, 0, 0.1, 80, 1, element_type="Quad9")
   problem = dualmesh.Problem(mesh)
   solid = problem.add_physics(
       "solid_mechanics",
       formulation="plane_stress",
       youngs_modulus=200.0e9,
       poissons_ratio=0.0,
       density=7850.0,
       initial_condition={"velocity": ["sin(pi*x/2)", 0]},
   )
   solid.add_boundary_condition("disp_x = 0", boundary="left")
   solid.add_boundary_condition("disp_y = 0", boundary=["left", "top", "bottom", "right"])
   problem.solve_transient(end_time=1.0e-4, time_step=1.0e-6)

Control the time step
~~~~~~~~~~~~~~~~~~~~~

``time_stepper`` selects how the step :math:`\Delta t` changes after the
first one.  The input names of ``step_control.by`` are in brackets.

``"error"`` (``error``)
   The default of ``solve_transient`` and of the input.  The step keeps an
   estimate of the local
   truncation error of a step at ``error_tolerance`` (default
   :math:`10^{-3}`).  For the :math:`\theta` method, the estimate compares the
   step with the polynomial through the last accepted states, extrapolated to
   the end of the step: linear for an order-one member, quadratic for
   Crank-Nicolson.  The projection of a flow compares its step with the
   polynomial of its own order through the last accepted states.  This costs
   one solve per step.  The first steps, which have too few earlier states,
   and a problem with inertia take step doubling instead: the step is taken
   once with :math:`\Delta t` and again as two steps of :math:`\Delta t / 2`,
   and the difference of the two answers estimates the error
   [RichardsonGaunt1927]_.  This costs three solves per step and keeps the
   answer of the half steps.

   Each variable with a time derivative contributes the largest difference
   at a node between the two answers, divided by the scale of the variable,
   and the estimate :math:`e` is the largest contribution, so it has no unit.
   The scale is the largest of the range of the variable (its largest value
   less its smallest), its largest change from the initial state, and
   :math:`\sqrt{\epsilon_m}` times its largest magnitude, with
   :math:`\epsilon_m = 2^{-52}` the machine epsilon.  Each of the three is the
   largest it has reached in the run, and the components of a velocity or a
   displacement share the largest scale among them.  A temperature near
   :math:`300\ \mathrm{K}` is then measured against how far it has moved,
   a rigid motion or a uniform flow against its change, and a field at rest
   keeps its round-off below the tolerance.  A step whose estimate exceeds
   the tolerance :math:`\epsilon` is rejected.  After an accepted step of
   order :math:`p`, the next step is
   :math:`\Delta t \min\bigl(g, \max(1, 0.9\, (\epsilon/e)^{1/(p+1)})\bigr)`,
   with :math:`g` the ``growth_factor`` [HairerNorsettWanner1993]_.  A
   problem without a time derivative measures the change of every variable.

``"fixed"`` (``fixed``)
   The step stays at ``time_step``.  The last step shortens to end exactly at
   ``end_time``, and a step shortens to land on an output time.  A step that
   fails to converge is retried at ``cutback_factor`` times its size, and the
   steps after it grow by ``growth_factor`` each until they reach
   ``time_step`` again.  Use this stepper when the steps themselves matter,
   such as in a study of the convergence in the step.

``"iteration"`` (``iterations``)
   The step follows the number of nonlinear iterations of the last step.  A
   step that converged in fewer than ``optimal_iterations - iteration_window``
   iterations grows by ``growth_factor``, and a step that needed more than
   ``optimal_iterations + iteration_window`` shrinks by ``cutback_factor``.
   ``optimal_iterations`` (default 4, the input's ``target_iterations``) sets
   the target, and ``iteration_window`` (default 2)
   the half-width of the band.  It costs nothing beyond the solve and leaves
   the error in time uncontrolled, so it suits a problem whose difficulty is
   the nonlinearity.

``"cfl"``
   The Courant-Friedrichs-Lewy stepper keeps the Courant number
   :math:`C = \Delta t \max_e |\mathbf{v}| / h_e` at ``courant_number``
   (default 0.5), where the maximum runs over the elements :math:`e` and the
   speeds :math:`|\mathbf{v}|` at their nodes, and :math:`h_e` is the size of
   element :math:`e`.  Each new step gives that Courant number at the
   velocity that the last step reached, and grows by at most
   ``growth_factor``.  It applies to a flow advanced by the projection (see
   `Advance a flow by projection`_), which the input does not select, so the
   input has no name for it.

``growth_factor`` (default 2, at least 1) bounds the growth of the step from
one step to the next.  ``min_time_step`` (default ``time_step`` divided by
:math:`10^6`) and ``max_time_step`` (default the whole interval) bound the
step.

Rejected steps
~~~~~~~~~~~~~~

A step is rejected when its nonlinear solve does not converge, or, with the
``error`` stepper, when its estimate exceeds the tolerance.  The run restores
the state at the start of the step and takes the step again.  A step whose
solve failed is multiplied by ``cutback_factor`` (default 0.5, strictly between
0 and 1).  A step that the ``error`` stepper rejected is multiplied by
:math:`\max\bigl(0.1,\ 0.9\, (\epsilon/e)^{1/(p+1)}\bigr)`, the step that the
estimate allows, at least a tenth of the rejected one.

The run fails when a step would fall below ``min_time_step``, or when more
than ``max_rejected_steps`` (default 10) steps in a row are rejected.  It then
raises an error, or, with ``error_on_divergence=False``, returns with
``converged`` false.  ``result.rejected_steps`` counts the rejections, and
``result.step_history`` lists the time reached and the step taken for each
accepted step.

This steel slab, at :math:`300\ \mathrm{K}`, has one face held at
:math:`3000\ \mathrm{K}` and the other radiating to an ambient of
:math:`300\ \mathrm{K}`.  The radiation makes the first step hard to
converge, and the run starts with a step far too large for two iterations, so
the ``iteration`` stepper cuts it back (from
``tests/python/test_time_stepping.py``):

.. code-block:: python

   import dualmesh
   import numpy

   mesh = dualmesh.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
   problem = dualmesh.Problem(mesh)
   problem.add_variable("temperature")
   problem.set_values("temperature", numpy.full(mesh.num_nodes, 300.0))
   problem.add_kernel("diffusion", variable="temperature", diffusivity=20.0)
   problem.add_kernel("time_derivative", variable="temperature", coefficient=7800.0 * 460.0)
   problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=3000.0)
   # Grey-body radiation with the emissivity 0.9, as the inflow -e sigma (T^4 - T_a^4).
   problem.add_boundary_condition(
       "Neumann", "right", variable="temperature", flux="-5.670374419e-8*0.9*(temperature^4 - 300^4)"
   )
   result = problem.solve_transient(
       end_time=400.0,
       time_step=400.0,
       time_scheme="backward_Euler",
       time_stepper="iteration",
       max_iterations=2,
       cutback_factor=0.25,
   )
   print(result.converged, result.rejected_steps, len(result.step_history))

Start a flow
~~~~~~~~~~~~

A transient run of an incompressible flow first projects the initial velocity
onto a field without divergence that keeps the prescribed values, with the
discrete divergence of the flow's own method (see :doc:`../theory/fluids`).  The
correction is a discrete gradient, so a fluid at rest behind a prescribed
inflow starts from the irrotational flow that carries the inflow through the
domain, and the first steps take the size that the flow needs.  A field without divergence stays as it is, and so does the state
of a run that restarts from a checkpoint or goes on from where the last run
of the problem stopped.  The report of the solve gives the size of the
correction in the row ``initial velocity correction``: the largest change of
the velocity at a node over the largest speed after the projection.
``result.start_correction`` holds the same number, and ``None`` for a run that
projected nothing.

Advance a flow by projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fluid_dynamics`` advances a transient flow in one of two ways, which its
parameter ``time_integration`` selects:

``"monolithic"``
   The default.  Each step solves the velocity and the pressure together, by
   the nonlinear method of `Choose the nonlinear method`_ and with the time
   scheme of ``solve_transient``.

``"projection"``
   Each step solves a pressure Poisson equation, preconditioned with
   BoomerAMG, and then one Helmholtz equation per velocity component,
   preconditioned with Jacobi.  The time derivative takes the backward
   difference formula of order ``time_order`` (1, 2 or 3, default 2), and the
   advection takes the extrapolation of the same order.  The projection needs
   ``formulation="equal_order"`` and a positive density.  It runs with the
   finite element method (FEM), where the iterations are the conjugate
   gradient method, and with the dual mesh control domain method (DMCDM),
   where they are GMRES, on one process and on several.

This decaying Taylor-Green vortex in a periodic box keeps the Courant number
at one half (after ``tests/python/test_projection.py``):

.. code-block:: python

   import dualmesh
   import numpy

   mesh = dualmesh.generate_rectangle_mesh(0.0, 2 * numpy.pi, 0.0, 2 * numpy.pi, 64, 64)
   problem = dualmesh.Problem(mesh, method="FEM")
   problem.add_physics(
       "fluid_dynamics",
       density=1.0,
       dynamic_viscosity=0.01,
       formulation="equal_order",
       time_integration="projection",
   )
   problem.add_boundary_condition("periodic", primary="left", secondary="right")
   problem.add_boundary_condition("periodic", primary="bottom", secondary="top")
   problem.initialize()
   points = numpy.asarray(mesh.points())
   x, y = points[:, 0], points[:, 1]
   problem.set_values("velocity_x", -numpy.cos(x) * numpy.sin(y))
   problem.set_values("velocity_y", numpy.sin(x) * numpy.cos(y))
   result = problem.solve_transient(
       end_time=10.0, time_step=0.01, time_stepper="cfl", courant_number=0.5
   )

The pressure of an enclosed flow is fixed up to a constant, and the solver
reports it with a mean of zero over the fluid.

When a solve fails
------------------

Each error message states the cause of the failure and what to change.  The
common messages are these:

*"nonlinear solve did not converge within ... iterations"*
   The residual stopped falling.  Look at ``result.history``.  A residual
   that falls and then flattens points to an inexact Jacobian, which happens
   when a kernel of your own computes a quantity as a plain ``float`` and so
   drops the derivatives that a :class:`~dualmesh.ADReal` carries.

*"Newton's method diverged: the residual grew ..."*
   The iterate left the region where Newton's method converges.  A steady
   solve has tried continuation already.  Start from a closer state, refine
   the mesh, or use ``nonlinear_solver="Picard"`` with ``relaxation``.

*"the line search found no step length ..."*
   No fraction of the Newton step, down to :math:`2^{-k}` with :math:`k` the
   value of ``max_line_search_steps``, lowers the residual.  A property with a
   kink, a singular Jacobian or a problem without a solution causes this.

*"the residual or the Jacobian has NaN or infinite entries"*
   Either Newton's method diverged or a property or its derivative is not
   finite at the current solution, for example :math:`\log 0`.

*"the linear solve did not converge"*
   The solver that ``petsc_options`` chose, or the last solver of the
   automatic choice, did not reach its tolerance.  The
   message gives PETSc's reason and the iterations taken.  Leave out the
   options of the linear solver for the automatic choice, or give a direct
   solver, ``petsc_options="-ksp_type preonly -pc_type lu"``.

*"The system is singular: the variable ... is not held by any boundary condition"*
   A group of coupled unknowns has no condition that fixes its level, so a
   constant (a temperature, a pressure) or a rigid body motion can be added
   to it without changing the equations.  A body without supports, a region
   with no temperature condition that exchanges no heat with the rest, and an
   imported mesh whose parts do not share their nodes cause this.  The message
   gives the number of nodes and the extent of the part.  A Robin condition,
   a reaction term or a time derivative holds the level, and the check takes
   them into account.

*"the transient solve failed at t = ..."*
   The step was rejected ``max_rejected_steps`` times in a row or reached
   ``min_time_step``.  Some problems have no bounded solution, so check the
   physics before you lower ``min_time_step``.

*"the solution has NaN or infinite values after the step to t = ..."*
   Either the step is too large for the explicit terms (lower ``time_step``
   or ``courant_number``), or a property is not finite at the solution.

*"degenerate element (zero Jacobian determinant)"*
   An element has a zero or inverted volume.  ``mesh.fix_orientation()``
   repairs a consistently inverted mesh.  Check that a
   ``mesh.transform_nodes`` map does not fold the mesh over itself.

*"Too many variables for this mesh"*
   The automatic differentiation has a slot for each degree of freedom of an
   element, up to a limit.  The message gives the count and the CMake setting
   ``DUALMESH_MAX_AD_DERIVATIVES`` that raises the limit.

Run in parallel
---------------

Threads
~~~~~~~

The assembly loops run on several threads by OpenMP (Open Multi-Processing).
:meth:`~dualmesh.Problem.set_num_threads` sets the number of threads, where 0,
the default, means the OpenMP default: the value of the ``OMP_NUM_THREADS``
environment variable, or the number of cores.  The assembly runs on one
thread when the library was built without OpenMP, when the mesh is too small
for threads to pay for themselves, or when any kernel, boundary condition,
material or function of the problem is defined in Python.
:meth:`~dualmesh.Problem.effective_threads` reports the number the assembly
uses.

Processes
~~~~~~~~~

Run a script or a case file on several processes of the Message Passing
Interface (MPI) with ``mpiexec``:

.. code-block:: bash

   mpiexec -n 4 python my_analysis.py
   mpiexec -n 4 dualmesh run case.yaml

Every process runs the same script and defines the same physics and objects.
A :class:`~dualmesh.Problem` splits itself among the processes when the run
has more than one, and each process assembles the elements of its own part of
the mesh.  The solvers on this page run the same way in a distributed problem.
The parameters of :class:`~dualmesh.Problem` that set the split are:

``distributed``
   ``None``, the default, splits the problem when the run has more than one
   process.  ``True`` takes the distributed path on one process as well, and
   ``False`` keeps a whole problem on each process.

``partitioner``
   The partitioner of PETSc that splits the elements: ``"automatic"`` (the
   default: PT-Scotch, else ParMETIS), ``"ptscotch"``, ``"parmetis"`` or
   ``"simple"``.

``gather_scatter``
   The library that exchanges the values at the nodes that several processes
   hold: ``"petsc"`` (the default, PETSc's star forests) or ``"gslib"`` (the
   gather-scatter library of Nek5000).  The answer is the same with either.

:meth:`~dualmesh.Problem.gathered_values` returns the values of the whole mesh
on every process.  :meth:`~dualmesh.Problem.write_vtu` writes a distributed
problem as one ``.vtu`` file per process and a ``.pvtu`` index, which ParaView
opens as one data set.
