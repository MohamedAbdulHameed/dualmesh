Solving
=======

Once the variables, kernels, boundary conditions and property objects are in place, the
problem is solved either once, for a steady state, or repeatedly in time.  This
chapter explains both, and every option each takes.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

The steady solve
----------------

.. code-block:: python

   result = problem.solve()
   print(result.converged, result.total_iterations)

:meth:`~dualmesh.Problem.solve` assembles the residual and the Jacobian,
solves the linear system, and repeats until one of the convergence tests below
passes.  SNES, the nonlinear solver of PETSc [PETSc2023]_, runs this
iteration with the convergence test and the line search described below.  It
returns a :class:`~dualmesh.SolveResult` and leaves the solution on the problem,
where :meth:`~dualmesh.Problem.values` reads it out.

All options are keyword arguments.  An unknown keyword is an error that lists
the ones that are known, so a typo does not become a silently ignored setting.

Choosing the nonlinear method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nonlinear_solver`` takes three values.

``"Newton"``
   The default.  At each iteration the tangent matrix
   :math:`J = \partial R / \partial U` is assembled and the correction
   :math:`\delta U = -J^{-1} R` is applied.  The tangent is *exact*, because
   every kernel is evaluated in forward-mode automatic differentiation
   [Wengert1964]_, which requires neither hand-derived derivatives nor
   finite-difference approximations.  The iteration therefore converges
   quadratically: the number of correct digits doubles at each step once the
   iterate is close enough to the solution.  The quadratic convergence can be
   observed in ``result.history``.

``"Picard"``
   Direct iteration, also called successive substitution.  The nonlinear
   coefficients are evaluated at the previous iterate, the resulting linear
   problem is solved, and the solution becomes the next iterate.  It converges
   linearly, while Newton's method converges quadratically, but each iteration
   is cheaper and its basin of attraction is often wider, so it is the usual
   alternative when Newton's method diverges from a poor starting point.  The
   ``relaxation`` option described below applies to this method.

``"linear"``
   Assemble once, solve once, and check the residual of the result.  Selecting
   this option asserts that the problem is linear: if the residual after the
   one solve is above ``relative_tolerance``, the solve stops with an error,
   because the problem is nonlinear or the linear solver did not converge.

Convergence and its tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The iteration stops when the residual passes the test below.  The test has no
unit, so a problem converges in the same iterations in metres and in
millimetres (see :doc:`/theory/nonlinear_and_time`).

``relative_tolerance`` (default :math:`10^{-10}`)
   The residual of every equation (the unknowns of one variable) is at most
   this fraction of the size of its own terms, the reference residual.  A step
   that starts almost converged needs no more iterations than that.

``max_iterations`` (default 50) limits the number of iterations.
``error_on_divergence`` (default true) determines what happens when this limit
is reached: with it true an exception is raised, and with it false the result
is returned with ``converged`` false so that the caller can decide.  Set it
false inside a loop that is expected to fail sometimes, such as a continuation
study, and leave it true otherwise.

``relaxation`` applies to an iteration without a line search (direct iteration,
or Newton's method without one) and defaults to zero, meaning no relaxation.  With :math:`\gamma` set, the coefficients are evaluated at the
weighted combination of the last two iterates

.. math::

   \bar{U} = (1 - \gamma) U^{(r)} + \gamma U^{(r-1)} ,

so each update takes the fraction :math:`1 - \gamma` of its step.  A value from
zero to below one under-relaxes and damps an oscillating iteration, and the code
refuses any other.  Under-relaxation is the usual remedy for a direct iteration
that oscillates without settling.

The full report (``report="full"``, the default) prints the residual norm at
every iteration. The residual norms show whether an iteration converges slowly
or stalls.

Continuation
~~~~~~~~~~~~

A nonlinear problem that does not converge from its starting state often
converges if its data are applied in increments, with each solve starting from
the solution of the previous one.  A steady solve does this by itself when
Newton's method fails on the problem itself: it ramps the sources, body forces,
prescribed fluxes and loads up from zero, and the prescribed values from those
of the starting state, while the terms that represent stiffness, conduction and
viscosity keep their full value.  A step that converges doubles the next one and
a step that fails is halved.  This is how a flow at a high Reynolds number, a
strongly nonlinear material and the von Kármán plates and beams are driven to
their solution, with no setting.  ``result.history`` records each step, with its
``continuation_step`` and ``data_fraction``.

.. _linear-solver-choice:

The linear solver
~~~~~~~~~~~~~~~~~

PETSc [PETSc2023]_ solves each linearised system :math:`J \, \delta U = -R`,
where :math:`J` is the Jacobian, :math:`\delta U` the Newton update and
:math:`R` the residual.  A serial problem and a distributed one take the same
path.  Without options, the solver chooses a method from the structure of the
system.  ``petsc_options`` replaces that choice with PETSc's own options.

The automatic choice
   The solver factorises the system directly where the factorisation is
   inexpensive: every one-dimensional system, two-dimensional systems up to
   :math:`10^5` unknowns, and three-dimensional systems up to 4000 unknowns.
   The factorisation is LU with partial pivoting [Demmel1999]_ on one process
   and the parallel direct solver MUMPS [Amestoy2001]_ on several.  Pivoting
   lets it factorise the saddle point systems of the Taylor-Hood element,
   which have zeros on the diagonal.

   These limits follow from the fill-in of a sparse factorisation.  With a
   good ordering, the factors of a two-dimensional mesh problem with :math:`n`
   unknowns hold :math:`O(n \log n)` entries and cost :math:`O(n^{3/2})`
   operations.  The factors of a three-dimensional problem hold
   :math:`O(n^{4/3})` entries and cost :math:`O(n^2)` operations
   [George1973]_ [LiptonRoseTarjan1979]_.  A preconditioned Krylov iteration
   costs a few matrix-vector products of :math:`O(n)` operations each per
   iteration.

   A larger system takes the iteration that suits its structure:

   - A flow takes flexible GMRES (the generalised minimal residual method
     [Saad1993]_) with the Schur complement split and the least squares
     commutator, ``-dualmesh_schur lsc`` below.
   - A solid takes GMRES with smoothed aggregation multigrid (GAMG) and the
     rigid body modes, described below.
   - A solid coupled to other physics takes GMRES with the split by physics,
     ``-dualmesh_split physics`` below.
   - A system whose variables have elements of different orders, other than a
     flow, is factorised.
   - Any other system takes GMRES with the algebraic multigrid of hypre,
     BoomerAMG [FalgoutYang2002]_.

   When the iteration does not converge in 500 iterations, the direct
   factorisation follows, from the iterate where the iteration stopped.  The
   automatic choice is therefore at least as robust as the direct solver.
   The iterations are preconditioned from the right, so the residual they
   measure is the true residual of the system.

   Within Newton's method, the automatic iterations solve each linear system
   only as accurately as the current Newton step needs.  The tolerance of
   Eisenstat and Walker [EisenstatWalker1996]_ is loose while the residual is
   large and tightens as it falls, which keeps the quadratic convergence of
   Newton's method and saves Krylov iterations in its first steps.  A direct
   solve, a solver chosen with ``petsc_options`` and the solve of
   ``nonlinear_solver="linear"`` take ``linear_tolerance`` instead.

``petsc_options``
   PETSc's own options, as the string PETSc parses or as a dictionary, choose
   the Krylov method and the preconditioner:

   .. code-block:: python

      problem.solve(petsc_options="-ksp_type preonly -pc_type lu")
      problem.solve(petsc_options={"ksp_type": "cg", "pc_type": "gamg"})

   ``linear_tolerance`` (default :math:`10^{-12}`, relative to the right-hand
   side) is passed on as PETSc's ``ksp_rtol``, and the options may override
   it.  The options that start with ``-snes_`` go to SNES, PETSc's nonlinear
   solver, which runs Newton's method, for example ``-snes_monitor`` or
   ``-snes_ksp_ew`` (the tolerance of Eisenstat and Walker with a solver of
   your choice).  The test of convergence stays that of
   ``relative_tolerance``.  The options of one solve are kept in a private options database, so
   they apply to that solve alone.  A solve that does not converge raises an
   error that gives PETSc's reason, ``DIVERGED_ITS`` for instance.  The input
   file takes the same options under ``solver: PETSc_options``.

   The unknowns are numbered node by node, all variables of a node together,
   and the number of variables is given to PETSc as the block size of the
   matrix.  PETSc's field split preconditioners need this numbering.  For the
   pressure-velocity formulation of a two-dimensional flow, whose variables
   are ``u``, ``v`` and ``pressure`` in that order, the Schur complement
   preconditioner with algebraic multigrid on both blocks is

   .. code-block:: python

      options = (
          "-ksp_type fgmres"
          " -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2"
          " -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper"
          " -pc_fieldsplit_schur_precondition selfp"
          " -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type hypre"
          " -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre"
      )

   On the cavity at :math:`Re = 100` on a :math:`64 \times 64` mesh it takes
   about 58 iterations per Newton step, on one process and on four, where
   :math:`Re` is the Reynolds number.  The same options converge for the
   Taylor-Hood element, whose pressure block is zero, but more slowly.  On the
   same cavity with :math:`32 \times 32` ``Quad9`` elements, which have the
   nodes of the :math:`64 \times 64` linear mesh, they take about 134
   iterations per Newton step at a relative tolerance of :math:`10^{-8}`,
   compared with 66 for the equal-order element at the same tolerance.  The
   ``selfp`` approximation of the Schur complement,
   :math:`\mathbf{B}\, \mathrm{diag}(\mathbf{A})^{-1} \mathbf{B}^T`, suits
   the stabilised element better.  For both elements the Schur splits
   described next, which build their approximation of the Schur complement
   from the flow, converge in fewer iterations.

``-dualmesh_schur lsc``, ``pcd`` or ``pressure_mass``
   The Schur complement split of a pressure-velocity flow, an option of
   dualmesh-multiphysics that PETSc reads among its own.  The Jacobian is a
   saddle point matrix

   .. math::

      \begin{bmatrix} \mathbf{F} & \mathbf{G} \\ \mathbf{D} & \mathbf{C} \end{bmatrix}

   with the momentum block :math:`\mathbf{F}`, the pressure gradient
   :math:`\mathbf{G}`, the discrete divergence :math:`\mathbf{D}` and the
   pressure block :math:`\mathbf{C}`.  The pressure block is zero for the
   Taylor-Hood element and the pressure Laplacian of the stabilisation for
   the equal-order element.  The preconditioner is the block upper triangle

   .. math::

      \mathbf{P} = \begin{bmatrix} \mathbf{F} & \mathbf{G} \\ \mathbf{0} &
      \hat{\mathbf{S}} \end{bmatrix},

   with which GMRES converges in a number of iterations that depends only on
   how well :math:`\hat{\mathbf{S}}` approximates the Schur complement
   :math:`\mathbf{S} = \mathbf{C} - \mathbf{D}\mathbf{F}^{-1}\mathbf{G}`, a
   dense matrix.  Up to its sign, which is measured from the coupling blocks
   of the actual matrix,
   :math:`\mathbf{S} \approx \mathbf{G}^T \mathbf{F}^{-1} \mathbf{G}`.  The
   three splits differ in their approximation of it:

   ``lsc``, the least squares commutator [ElmanHowleShadidShuttleworthTuminaro2006]_:
      :math:`\hat{\mathbf{S}}^{-1} = \mathbf{A}_p^{-1} (\mathbf{G}^T
      \mathbf{Q}_v^{-1} \mathbf{F} \mathbf{Q}_v^{-1} \mathbf{G})
      \mathbf{A}_p^{-1}` with :math:`\mathbf{A}_p = \mathbf{G}^T
      \mathbf{Q}_v^{-1} \mathbf{G}` and :math:`\mathbf{Q}_v` the lumped
      velocity mass matrix.  It is built from the blocks of the matrix alone.
   ``pcd``, the pressure convection-diffusion preconditioner [KayLoghinWathen2002]_,
      in the form of [ElmanTuminaro2009]_:
      :math:`\hat{\mathbf{S}}^{-1} = \mathbf{A}_p^{-1} \mathbf{F}_p
      \mathbf{Q}_p^{-1}`, where :math:`\mathbf{F}_p` is the operator of the
      momentum equation on the pressure space,
      :math:`(\rho/\Delta t) \mathbf{M}_p + \theta(\mu \mathbf{K}_p + \rho
      \mathbf{N}_p(\mathbf{v}))` at the current velocity :math:`\mathbf{v}`,
      and :math:`\mathbf{Q}_p` is the lumped pressure mass matrix.  Here
      :math:`\rho` is the density, :math:`\mu` the dynamic viscosity,
      :math:`\Delta t` the time step, :math:`\theta` the weight of the time
      scheme, and :math:`\mathbf{M}_p`, :math:`\mathbf{K}_p` and
      :math:`\mathbf{N}_p` the pressure mass, diffusion and advection
      matrices.  :math:`\mathbf{F}_p` carries the Robin condition
      :math:`\mu\, \partial p/\partial n = \rho (\mathbf{v} \cdot
      \mathbf{n})\, p` on the boundaries where the velocity is prescribed.
      It acts at an inflow, since :math:`\mathbf{v} \cdot \mathbf{n} = 0` on
      a wall.
   ``pressure_mass``:
      :math:`\hat{\mathbf{S}} = \mathbf{C}_{\mathrm{sym}} \pm (\tfrac{1}{\mu}
      q, \psi)`, the pressure mass matrix scaled by the inverse viscosity and
      the symmetric part :math:`\mathbf{C}_{\mathrm{sym}}` of the pressure
      block.  For the Stokes equations it is spectrally equivalent to
      :math:`\mathbf{S}` with bounds that do not depend on the mesh size, but
      it ignores convection.

   For the stabilised equal-order element, :math:`\mathbf{G}^T
   \mathbf{Q}_v^{-1} \mathbf{G}` has the spurious pressure modes of the
   element in its null space.  :math:`\mathbf{A}_p` and the middle factor of
   ``lsc`` therefore take the element stabilisation of
   [ElmanHowleShadidSilvesterTuminaro2007]_: the :math:`L^2` distance of the
   pressure from its element mean, scaled by :math:`h^{-2}` and by
   :math:`\theta\mu h^{-4} + \rho/(\Delta t\, h^2)`, with :math:`h` the
   element size.  ``pcd`` applies :math:`\mathbf{A}_p^{-1}` once, to a vector
   that keeps the spurious modes at full size, and takes this stabilisation a
   hundred times larger.  Measured, the iteration counts are three to four
   times higher with the factor 1, and change by less than 15 % between the
   factors 100 and :math:`10^5`.  An assembled pressure Laplacian in place of
   :math:`\mathbf{A}_p` needs a Robin condition at an outflow whose weight
   changes the counts fivefold, which is why :math:`\mathbf{A}_p` is taken
   from the matrix.

   PETSc's Schur complement field split applies the approximation.  The
   velocities, and any other unknown such as a temperature, form the split
   ``0`` and the pressure the split ``1``.  The defaults are flexible GMRES
   outside, the upper block-triangular factorisation, and one V-cycle of
   BoomerAMG for :math:`\mathbf{F}` and for :math:`\mathbf{A}_p` (options
   prefix ``fieldsplit_1_laplacian_``), with the coarsening and the
   interpolation of [DeSterckYangHeys2006]_ and
   [DeSterckFalgoutNoltingYang2008]_.  With them the least squares
   commutator solves the three-dimensional Taylor-Hood cavity below 3.2 to 4.3
   times faster than with BoomerAMG's defaults, in twice the iterations.
   ``petsc_options`` may change any of them, for instance
   ``{"dualmesh_schur": "lsc", "fieldsplit_0_pc_type": "lu"}`` for an exact
   momentum solve.  With ``pcd`` and ``lsc`` the preconditioner of the split
   ``1`` is the approximation itself, so the options
   ``pc_fieldsplit_schur_precondition`` and ``fieldsplit_1_pc_type`` do not
   apply.  ``pcd`` needs an accurate momentum solve: on the
   three-dimensional cavity at :math:`Re = 400` with one V-cycle on
   :math:`\mathbf{F}` it does not converge in 5000 iterations.

   On a three-dimensional cavity at :math:`Re = 100` on two processes, with
   the default block solvers, the least squares commutator against MUMPS
   takes the following times for the whole steady solve of five Newton
   iterations:

   .. list-table::
      :header-rows: 1

      * - Element
        - Unknowns
        - MUMPS
        - ``lsc``
      * - Taylor-Hood (``Hex27``)
        - 37 044
        - 7.8 s
        - 3.9 s
      * -
        - 97 556
        - 22.2 s
        - 13.3 s
      * -
        - 143 748
        - 44.4 s
        - 23.0 s

   For the equal-order element (``Hex8``, 62 500 unknowns) it takes 8.1 s at
   :math:`Re = 100` and 20.8 s at :math:`Re = 400`, where BoomerAMG on the
   whole system with its defaults takes 29.9 s and 108 s.  BoomerAMG on the
   whole system with the coarsening of the block solvers takes 3.3 s at
   :math:`Re = 100` but stagnates at :math:`Re = 400`.

``-dualmesh_split physics``
   The split by physics, an option of dualmesh-multiphysics that PETSc reads
   among its own.  The unknowns of each physics form a block: the
   displacements of a solid together, and every other variable alone.  The
   block lower-triangular preconditioner solves the blocks one after the
   other, in the order in which the physics were added, each with GAMG on its
   diagonal block, and moves the coupling to the earlier blocks to the
   right-hand side.  A coupling that one physics drives, such as the thermal
   expansion of a solid by a temperature, is then exact in the
   preconditioner when that physics is added first.  On the heated plate of
   :doc:`/user_guide/case_files`, refined from :math:`100 \times 20` to
   :math:`400 \times 80` elements, it converges in 17 to 18 iterations at a
   relative tolerance of :math:`10^{-10}`.  A flow is refused: its velocity
   and pressure need ``-dualmesh_schur``, which takes any other unknown into
   its first block.

Smoothed aggregation with the rigid body modes
   PETSc's smoothed aggregation multigrid (GAMG) [VanekMandelBrezina1996]_
   builds a sequence of coarser problems from the matrix alone.  It groups
   the nodes into aggregates of strongly connected neighbours, and on each
   aggregate the coarse basis represents the *near null space* of the
   operator exactly.  The near null space consists of the vectors on which
   the operator nearly vanishes: the constant for a diffusion equation, and
   the three translations and three rotations of a rigid body (two and one in
   two dimensions) for elasticity.  dualmesh-multiphysics builds it from the
   displacement variables of the stress property objects of the problem and
   the constant of every other variable, and gives it to PETSc, also when
   ``petsc_options`` chooses GAMG.  GMRES is preconditioned from the right,
   so that it measures the true residual.  On a three-dimensional cantilever
   on two processes it takes 54 and 51 iterations on meshes of 4 and 8
   elements across, against 78 and 82 for BoomerAMG without the modes, and
   28 % less time.

For use with an external solver, :meth:`~dualmesh.Problem.linear_system` returns
the residual and the Jacobian of the current Newton step as a NumPy array and a
SciPy sparse matrix.

Reading the result
~~~~~~~~~~~~~~~~~~

A :class:`~dualmesh.SolveResult` carries:

``converged``
   Whether the iteration met one of the tolerances.

``total_iterations``
   Nonlinear iterations taken, summed over the continuation steps and time steps.

``linear_iterations``
   Krylov iterations taken, summed the same way.

``history``
   The residual norm at every nonlinear iteration.  It is the first diagnostic
   to examine when a solve fails.

``time_steps``, ``rejected_steps``, ``step_history``
   Filled by a transient solve (see below).

The transient solve
-------------------

.. code-block:: python

   result = problem.solve_transient(end_time=100.0, time_step=1.0, time_scheme="Crank-Nicolson")

Time is discretised with the :math:`\theta` method.  Writing :math:`M` for the
storage term and :math:`R_{\text{steady}}` for everything else, one step solves

.. math::

   M \frac{U^{n+1} - U^{n}}{\Delta t}
   + \theta R_{\text{steady}}(U^{n+1}, t^{n+1})
   + (1 - \theta) R_{\text{steady}}(U^{n}, t^{n}) = 0 .

where :math:`\theta` is set by ``time_scheme``, which selects the member of the
family.

:math:`\theta = 1`, ``time_scheme="backward_Euler"``
   Backward Euler, the default.  First-order accurate in time and
   unconditionally stable, and strongly damping, which makes it robust on a
   stiff problem and on the first step after a discontinuous start.

:math:`\theta = 1/2`, ``time_scheme="Crank-Nicolson"``
   Crank-Nicolson [CrankNicolson1947]_.  Second-order accurate and
   unconditionally stable, but only neutrally damping, so a sharp initial
   transient can produce oscillations that persist for several steps.

:math:`\theta = 0`, ``time_scheme="forward_Euler"``
   Forward Euler.  The steady terms are evaluated entirely at the old state, so
   the scheme is explicit and only conditionally stable.  For the slab problem
   of the test suite the critical step is between :math:`0.2 h^2` and
   :math:`0.3 h^2`, and above it the solution diverges within a few steps.

An equation without a time derivative, such as the mass equation of an
incompressible flow or the equilibrium of a quasi-static solid, is a
constraint of the new time, so every :math:`\theta` takes it at
:math:`t^{n+1}` only, with the weight 1, and so does the pressure gradient of
a flow.  The pressure is then the pressure of the step.  Weighted between the
two times, it would be an average of two steps, whose error comes back with the
opposite sign at every step, and a material with creep would be evaluated at
the old time without its history.  With :math:`\theta < 1` the pressure of a
step lies between the two times, so a pressure that the conditions prescribe
as a function of time is first-order accurate in time.

**Inertia.**  A problem with a second time derivative, such as a solid with a
``density``, takes ``time_scheme="generalized_alpha"`` by default, with the
damping of unresolved modes set by ``spectral_radius`` (0.75, COMSOL's
default; 1 gives none), or ``"Newmark"``, the average acceleration rule, which
keeps the energy:

.. code-block:: python

   solid = problem.add_physics(
       "solid_mechanics", formulation="plane_stress", youngs_modulus=200e9,
       poissons_ratio=0.3, density=7850.0,
       damping={"mass": 0.1, "stiffness": 1e-5},          # Rayleigh, C = a M + b K
       initial_condition={"velocity": ["sin(pi*x/2)", 0]},
   )
   problem.solve_transient(end_time=0.1, time_step=1e-4)
   modes = problem.solve_eigenvalue(num_modes=6)
   modes.frequencies  # Hz

``start_time`` (default zero) sets the initial time. An
:class:`~dualmesh.Output` group in the ``output`` parameter writes the fields at
the output times, and the time steps land on these times. For more
information, see :doc:`output`. The transient solve also takes all the steady
options, which apply to the nonlinear solve in each step.

Choosing the time step
~~~~~~~~~~~~~~~~~~~~~~

``time_stepper`` selects how :math:`\Delta t` is chosen after the first step.

``"fixed"``
   The default.  The step stays at ``time_step``, except that the last step is
   shortened so that the run ends exactly at ``end_time``.

``"error"``
   The step is chosen so that an estimate of the *local truncation error* stays
   below ``error_tolerance`` (default :math:`10^{-3}`).  The :math:`\theta`
   method estimates it from one step: the polynomial through the last accepted
   states, extrapolated to the end of the step, has an error of the same order
   as the step, so the difference between the two is in proportion to the error
   of the step.  The polynomial is linear for backward and forward Euler and
   quadratic for Crank-Nicolson.  The first steps, which have no earlier states
   to extrapolate, an equation with a second time derivative and the projection
   time integration of a flow take step doubling instead: the step is taken
   once with :math:`\Delta t` and again as two steps of :math:`\Delta t / 2`,
   and the difference between the two answers, scaled by the order of the
   method, estimates the error of the coarser one [RichardsonGaunt1927]_.  Step
   doubling costs three solves per step, and extrapolation one.  The difference
   of each variable is measured against its range (its largest value less its
   smallest), and the largest over the variables is the estimate, so it has no
   unit and a temperature offset does not hide it.  Only the variables with a
   time derivative count: a variable without one, such as the pressure of an
   incompressible flow, is algebraic and has no truncation error of its own.  The next step is then set
   by the standard controller of [HairerNorsettWanner1993]_.  It is the only
   stepper that measures the accuracy, so it is the stepper to use when the
   solution must meet a stated tolerance.  Tightening
   ``error_tolerance`` shortens the steps and reduces the error until the
   spatial discretisation error dominates, beyond which further tightening only
   increases the run time.

``"iteration"``
   The step is chosen from the number of nonlinear iterations that the last
   step needed.  A step that converged in fewer than
   ``optimal_iterations - iteration_window`` nonlinear iterations is followed
   by a longer one, and a step that needed more than
   ``optimal_iterations + iteration_window`` is followed by a shorter one.  It
   adds no cost beyond the solve itself.  It controls the computational
   *effort* and leaves the error uncontrolled, so it suits a nonlinear problem
   whose difficulty varies through the run and is unsuitable when accuracy is
   the concern.

``"cfl"``
   The step is chosen to keep the Courant number
   :math:`\Delta t \max_e |\mathbf{v}| / h_e` at ``courant_number`` (default
   0.5), as Nek5000 and nekRS choose their steps.  It applies to a flow advanced
   by the projection time integration (see `Transient flow`_ below).

``growth_factor`` (default 2.0) limits the growth of the step in one update,
and ``min_time_step`` and ``max_time_step`` set its lower and upper bounds.
``max_time_step`` at its default of zero means no upper bound.

Rejecting and retrying a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A step is rejected when its nonlinear solve fails to converge, or, with the
error stepper, when its estimated error exceeds the tolerance.  A rejected step
is discarded entirely (the state is restored to the beginning of the step) and
retried with the step multiplied by ``cutback_factor`` (default 0.5).

``max_rejected_steps`` (default 10) limits how many consecutive rejections are
allowed before the run stops, and ``min_time_step`` sets the other limit: a step that
has been cut back below it cannot be cut back again.  In either case the run
ends with an error stating that the transient solve failed, so that the step is
never reduced indefinitely.  ``result.rejected_steps`` counts the rejections,
and ``result.step_history`` lists every accepted ``(time, time_step)`` pair, whose plot
shows the behaviour of the step controller.

A worked example
~~~~~~~~~~~~~~~~

The example is a steel slab, initially at 300 K, with one face held at 3000 K
and the other radiating to an ambient of 300 K.  The radiation makes the problem
nonlinear and the first step difficult to converge, so the iteration stepper is
deliberately given an initial step that is far too large, and it reduces the
step itself.

.. code-block:: python

   import dualmesh
   import numpy

   mesh = dualmesh.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
   problem = dualmesh.Problem(mesh)
   problem.add_variable("temperature")
   problem.set_values("temperature", numpy.full(mesh.num_nodes, 300.0))

   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel(
       "heat_conduction_time_derivative", variable="temperature", density=7800.0, specific_heat=460.0
   )
   problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=3000.0)
   problem.add_boundary_condition(
       "radiative_heat_flux", "right", variable="temperature", emissivity=0.9, ambient_temperature=300.0
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

Transient flow
--------------

``fluid_dynamics`` advances a transient flow in one of two ways, chosen by
``time_integration``.  ``"monolithic"``, the default, solves the velocity and the
pressure together at every step, with the :math:`\theta` method above.
``"projection"`` is the splitting of Nek5000 and nekRS: every step solves a
pressure Poisson equation, by the conjugate gradient method with algebraic
multigrid, and then one Helmholtz equation per velocity component, by the
conjugate gradient method with a Jacobi preconditioner, with the backward
difference and the extrapolation of order ``time_order`` (1, 2 or 3, default 2).
It needs ``formulation="equal_order"``, works with both methods, runs in
serial and in parallel, and suits long transients at moderate Courant numbers,
where each of its steps costs a fraction of a monolithic step.  The theory and
the verification are in :doc:`../theory/heat_and_fluids`.

The example advances the decaying Taylor-Green vortex in a periodic box, at a
Courant number of one half:

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
   x, y = numpy.asarray(mesh.points())[:, 0], numpy.asarray(mesh.points())[:, 1]
   problem.set_values("velocity_x", -numpy.cos(x) * numpy.sin(y))
   problem.set_values("velocity_y", numpy.sin(x) * numpy.cos(y))
   result = problem.solve_transient(
       end_time=10.0, time_step=0.01, time_stepper="cfl", courant_number=0.5
   )

The pressure of an enclosed flow is determined up to a constant, and the solver
fixes the constant itself: the pressure it reports has a mean of zero over the
fluid.

When a solve fails
------------------

Each error message states both the cause of the failure and the action that
corrects it.  The most common messages are the following.

*"Newton did not converge"*
   The residual stopped falling.  Look at ``result.history``: a residual that
   falls and then flattens usually indicates an inconsistent Jacobian, which in
   this library means that a kernel computed a quantity as a plain ``float``
   and so discarded the derivatives that a :class:`~dualmesh.ADReal` carries.
   A residual that grows means the iterate has left the basin of attraction,
   A steady solve has then tried continuation already; the remedies are a
   better initial condition, a finer mesh, or ``nonlinear_solver="Picard"``
   with under-relaxation.

*"the linear solve did not converge"*
   The iteration chosen by ``petsc_options`` did not reach ``linear_tolerance``.
   The message gives PETSc's reason and the iterations taken.  Leave out
   ``petsc_options`` for the automatic choice, or give a direct solver,
   ``petsc_options="-ksp_type preonly -pc_type lu"``.

*"The system is singular: the variable ... on a part of the mesh ... is not held by any boundary condition"*
   A group of coupled unknowns has no condition that fixes its level, so
   that a constant (a temperature, a pressure) or a rigid body motion can be
   added to it without changing the equations, and the solution is not
   unique.  Typical causes are a body without supports, a region that has no
   temperature condition and exchanges no heat with the rest of the problem,
   and an imported mesh whose parts do not share their nodes.  The message
   gives the number of nodes and the extent of the part.  Before the first
   linear solve, the code splits the Jacobian into its groups of coupled
   unknowns and applies it to the near null space vectors of each group (see
   the smoothed aggregation above), so that a Robin condition, a reaction term or a time
   derivative, which make the system nonsingular, never trigger the message.

*"transient solve failed"*
   The step was cut back to ``min_time_step`` or rejected ``max_rejected_steps`` times
   in a row and still did not converge.  Some problems have no bounded solution,
   in which case no step size leads to convergence, so check the physics before
   lowering ``min_time_step`` any further.

*"degenerate element (zero Jacobian determinant)"*
   An element has zero or inverted volume.  Call ``mesh.fix_orientation()``,
   which repairs a consistently inverted mesh, and check any
   ``transform_nodes`` that may have folded the mesh over itself.

*"Too many variables for this mesh"*
   The automatic differentiation budget is exhausted.  The message gives the
   arithmetic and the CMake setting that raises the limit.

Running in parallel
-------------------

Both solves thread their assembly loops automatically.
:meth:`~dualmesh.Problem.set_num_threads` sets the count and
:meth:`~dualmesh.Problem.effective_threads` reports what will actually be used,
which is one whenever any part of the problem is defined in Python.  A script
run with ``mpirun`` on several processes splits the problem among them, and
both forms of parallelism are described in :doc:`/theory/parallel`.
