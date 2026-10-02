Solving
=======

Once the variables, kernels, boundary conditions and materials are in place, the
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
passes.  It
returns a :class:`~dualmesh.SolveResult` and leaves the solution on the problem,
where :meth:`~dualmesh.Problem.values` reads it out.

All options are keyword arguments.  An unknown keyword is an error that lists
the ones that are known, so a typo does not become a silently ignored setting.

Choosing the nonlinear method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nonlinear_solver`` takes three values.

``"newton"``
   The default.  At each iteration the tangent matrix
   :math:`J = \partial R / \partial U` is assembled and the correction
   :math:`\delta U = -J^{-1} R` is applied.  The tangent is *exact*, because
   every kernel is evaluated in forward-mode automatic differentiation
   [Wengert1964]_, which requires neither hand-derived derivatives nor
   finite-difference approximations.  The iteration therefore converges
   quadratically: the number of correct digits doubles at each step once the
   iterate is close enough to the solution.  The quadratic convergence can be
   observed in ``result.history``.

``"picard"``
   Direct iteration, also called successive substitution.  The nonlinear
   coefficients are evaluated at the previous iterate, the resulting linear
   problem is solved, and the solution becomes the next iterate.  It converges
   linearly, while Newton's method converges quadratically, but each iteration
   is cheaper and its basin of attraction is often wider, so it is the usual
   alternative when Newton's method diverges from a poor starting point.  The
   ``relaxation`` option described below applies to this method.

``"linear"``
   Assemble once and solve once, with no iteration.  Use it when the problem is
   known to be linear, because it saves the one extra residual evaluation that
   Newton's method spends confirming convergence.  If the problem is in fact
   nonlinear, the result is the first Newton step and is wrong, so selecting
   this option asserts that the problem is linear.

Convergence and its tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The iteration stops when any of three tests passes.

``relative_tolerance`` (default :math:`10^{-10}`)
   The norm of the residual has fallen by this factor from its value at the
   first iteration.  This is the usual test, and it is scale free.

``absolute_tolerance`` (default :math:`10^{-12}`)
   The norm of the residual is below this value.  This test covers the case of
   a very small initial residual, for which the relative test would demand an
   unattainable precision.

``step_tolerance`` (default :math:`10^{-12}`)
   The norm of the correction is below this value.  In a problem whose residual
   cannot be reduced further, e.g., because of a badly scaled equation, the
   iterate may still have stopped changing, and this test detects that state.

``max_iterations`` (default 50) limits the number of iterations.
``error_on_divergence`` (default true) determines what happens when this limit
is reached: with it true an exception is raised, and with it false the result
is returned with ``converged`` false so that the caller can decide.  Set it
false inside a loop that is expected to fail sometimes, such as a continuation
study, and leave it true otherwise.

``relaxation`` applies to direct iteration only and defaults to zero, meaning no
relaxation.  With :math:`\gamma` set, the coefficients are evaluated at the
weighted combination of the last two iterates

.. math::

   \bar{U} = (1 - \gamma) U^{(r)} + \gamma U^{(r-1)} ,

so a value between zero and one under-relaxes and damps an oscillating
iteration, while a value above one over-relaxes and accelerates a slow
monotone one.  Under-relaxation is the usual remedy for a direct iteration that
oscillates without settling.

``verbose`` prints the residual norm at every iteration, which distinguishes a
slow convergence from a stalled one.

Load stepping
~~~~~~~~~~~~~

A nonlinear problem that does not converge from a zero initial guess often
converges if the load is applied in increments, with each solve starting from
the solution of the previous one.

.. code-block:: python

   result = problem.solve(load_factors=[0.25, 0.5, 0.75, 1.0])

Each factor multiplies every object that was added with
``scale_with_load=True``, so the terms that represent applied load are ramped
while the terms that represent stiffness keep their full value.  The factors
need not be evenly spaced, and the last one should normally be 1.0.  This is
how the von Kármán plate and beam problems of the solid mechanics module are
driven to large deflection.

.. _linear-solver-choice:

The linear solver
~~~~~~~~~~~~~~~~~

``linear_solver`` chooses how each linearised system :math:`J \, \delta U =
-R` is solved.  The default, ``"automatic"``, is the appropriate choice for
almost every problem, and the others are provided for control and for
comparison.

``"automatic"``
   The default.  It factorises the system directly where the factorisation is
   inexpensive and iterates otherwise.  If the iteration fails, it reverts to
   the direct solver, so that it is at least as robust as ``"lu"``.

   The threshold between the two follows from the fill-in of a sparse
   factorisation.
   With a good ordering, the factors of a two-dimensional mesh problem with
   :math:`n` unknowns hold :math:`O(n \log n)` entries and cost
   :math:`O(n^{3/2})` operations, but those of a three-dimensional problem hold
   :math:`O(n^{4/3})` entries and cost :math:`O(n^2)` operations
   [George1973]_ [LiptonRoseTarjan1979]_.  A preconditioned Krylov iteration
   costs a few matrix-vector products, each :math:`O(n)`, per iteration.
   ``"automatic"`` therefore factorises every one-dimensional system, two-dimensional
   systems up to :math:`10^5` unknowns, and three-dimensional systems up to a
   few thousand, and otherwise runs BiCGSTAB with an ILU(0) preconditioner.

   The difference is large in three dimensions.  For the three-dimensional
   elasticity problem of the verification suite on a :math:`12^3`-cell
   tetrahedral mesh, the cell-centred method has 36 288 unknowns and 1.3
   million matrix entries.  The direct solver takes 18 s (SuperLU, through
   SciPy, with the same column ordering, stores 47 times as many entries in
   the factors as in the matrix).  BiCGSTAB with ILU(0) reaches a relative
   residual of :math:`3 \times 10^{-11}` in 33 iterations and 0.3 s.

``"lu"``
   A sparse LU factorisation with partial pivoting [Demmel1999]_.  It is
   direct, so it has no tolerance to set, and it works for any nonsingular
   matrix.  Its cost and memory grow faster than the problem size, as above.

``"bicgstab"``
   The stabilised biconjugate gradient method [VanDerVorst1992]_.  It works for
   a general matrix.

``"gmres"``
   The generalised minimal residual method, restarted every ``gmres_restart``
   (default 60) iterations.  It works for a general matrix, and its residual
   never increases from one iteration to the next, which makes it the more
   robust of the two general methods on a difficult system.

``"cg"``
   The conjugate gradient method [HestenesStiefel1952]_, about half the cost of
   BiCGSTAB per iteration.  **It requires a symmetric positive definite matrix.**
   The Galerkin finite element method produces one.  The dual mesh control
   domain method and both finite volume methods produce nonsymmetric matrices,
   because a control volume balance is not a weighted residual statement and
   the coupling from node :math:`I` to node :math:`J` need not match the
   coupling from :math:`J` to :math:`I`.  Selecting ``"cg"`` with one of those
   methods produces a failure to converge, and the error message states this
   cause explicitly.

The Krylov methods take ``linear_tolerance`` (default :math:`10^{-12}`, relative
to the right-hand side), ``linear_max_iterations`` (default 5000) and
``preconditioner``:

``"ilu"``
   The default: the incomplete LU factorisation without fill, ILU(0)
   [Saad2003]_.  Its factors have exactly the sparsity pattern of the matrix, so
   it costs about as much as a few matrix-vector products to build and no more
   memory than the matrix.

``"ilut"``
   The threshold incomplete LU factorisation [Saad1994]_, which keeps fill
   entries larger than a drop tolerance (:math:`10^{-4}`, at most ten times the
   entries of a row).  It needs fewer iterations than ILU(0) but costs far more
   to build in three dimensions.

``"jacobi"``, ``"none"``
   Division by the diagonal, and no preconditioning.  Both are provided for
   comparison.

``"petsc"``
   The system is passed to PETSc [PETSc2023]_, which must have been found when
   the extension was built (see ``dualmesh.have_petsc()`` and
   :doc:`../installation`).
   PETSc is chosen for its preconditioners: algebraic multigrid
   (``pc_type`` ``"gamg"``, or ``"hypre"`` for BoomerAMG [FalgoutYang2002]_),
   the parallel direct solver MUMPS [Amestoy2001]_, and the field-split
   preconditioners of coupled problems.  Its Krylov method and preconditioner
   are chosen with PETSc's own options, passed as ``petsc_options``, either as
   the string PETSc parses or as a dictionary:

   .. code-block:: python

      problem.solve(linear_solver="petsc",
                    petsc_options={"ksp_type": "cg", "pc_type": "gamg"})

   Without options the default is GMRES preconditioned by a sparse LU
   factorisation, which converges in one iteration and so behaves like
   ``"lu"``.  The factorisation is done by MUMPS when PETSc has it, because
   MUMPS pivots and so also factorises saddle point systems with zeros on the
   diagonal, such as those of the Taylor-Hood element.  PETSc's own LU, which
   is used when MUMPS is unavailable, does not pivot.  ``linear_tolerance`` and
   ``linear_max_iterations`` are passed on as PETSc's ``ksp_rtol`` and
   ``ksp_max_it``, and the options may override them.  The options of one
   solve are kept in a private options database, so they apply to that solve
   alone.  A solve that does not
   converge raises an error that gives PETSc's reason (``DIVERGED_ITS``, for
   instance).

   The unknowns are numbered node by node, all variables of a node together,
   and the number of variables is given to PETSc as the block size of the
   matrix.  The field-split preconditioners require this numbering.  For the
   pressure-velocity formulation of a two-dimensional flow, whose variables
   are ``u``, ``v`` and ``pressure`` in that order, the Schur complement
   preconditioner with algebraic multigrid on both blocks is

   .. code-block:: python

      options = ("-ksp_type fgmres -pc_type fieldsplit "
                 "-pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 "
                 "-pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper "
                 "-pc_fieldsplit_schur_precondition selfp "
                 "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type hypre "
                 "-fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre")

   On the cavity at :math:`Re = 100` on a :math:`64 \times 64` mesh it takes
   about 58 iterations per Newton step, on one process and on four.  The same
   options also converge for the Taylor-Hood element
   (``formulation="taylor_hood"``), whose pressure block is zero (its inactive
   mid-side pressure slots are identity rows of that block and do not disturb
   the split), but more slowly: on the same cavity with :math:`32 \times 32`
   ``Quad9`` elements, which have the nodes of the :math:`64 \times 64` linear
   mesh, they take about 134 iterations per Newton step at a relative
   tolerance of :math:`10^{-8}`, compared with 66 for the equal-order element
   at the same tolerance.  The ``selfp`` approximation of the Schur complement,
   :math:`\mathbf{B}\, \mathrm{diag}(\mathbf{A})^{-1} \mathbf{B}^T`, is better
   suited to the stabilised element.  For both elements the pressure mass
   matrix preconditioner described next converges in fewer iterations.

``preconditioner="pressure_mass_schur"``
   For pressure-velocity flow.  The Jacobian is a saddle point matrix

   .. math::

      \begin{bmatrix} \mathbf{A} & \mathbf{G} \\ \mathbf{D} & \mathbf{C} \end{bmatrix}

   (momentum block, pressure gradient, discrete divergence, and the pressure
   block, which is zero for the Taylor-Hood element and the pressure
   Laplacian of the stabilisation for the equal-order element).  Its Schur
   complement :math:`\mathbf{S} = \mathbf{C} - \mathbf{D}\mathbf{A}^{-1}\mathbf{G}`
   is dense, but for the Stokes equations it is spectrally equivalent to the
   pressure mass matrix scaled by the inverse viscosity, with bounds that do
   not depend on the mesh size.  The preconditioner is the block upper
   triangle

   .. math::

      \mathbf{P} = \begin{bmatrix} \mathbf{A} & \mathbf{G} \\ \mathbf{0} &
      \hat{\mathbf{S}} \end{bmatrix}, \qquad
      \hat{\mathbf{S}} = \mathbf{C}_{\mathrm{sym}} \pm \left(\tfrac{1}{\mu} q, \psi\right),

   where the mass matrix is assembled with the pressure's own shape
   functions (the corner functions of a Taylor-Hood element), the sign is
   measured from the coupling blocks of the actual matrix, and
   :math:`\mathbf{C}_{\mathrm{sym}}` is the symmetric part of the pressure
   block.  With ``linear_solver="gmres"`` it runs in a built-in flexible
   GMRES with exact factorisations of :math:`\mathbf{A}` and
   :math:`\hat{\mathbf{S}}`.  With ``linear_solver="petsc"`` it sets up PETSc's
   Schur complement field split with :math:`\hat{\mathbf{S}}` as the
   preconditioning matrix, and ``petsc_options`` can replace the block
   solvers, for instance ``"-fieldsplit_0_pc_type gamg"`` for algebraic
   multigrid on the momentum block.

   Measured on the lid-driven cavity, iterations per Newton step at a
   relative tolerance of :math:`10^{-12}`:

   .. list-table::
      :header-rows: 1

      * - Element, flow
        - :math:`8 \times 8`
        - :math:`16 \times 16`
        - :math:`32 \times 32`
      * - Taylor-Hood (``Quad9``), Stokes
        - 15
        - 16
        - 16
      * - Stabilised equal order (``Quad4``), Stokes
        - 16
        - 18
        - 19

   At :math:`Re = 100` the counts rise to about 40, because the Schur
   complement then also carries convection, which the mass matrix does not
   describe.  In three dimensions the preconditioner also reduces the solution
   time: for Stokes flow on :math:`10^3` ``Tet10`` elements (29 000 unknowns) a
   direct solve takes 21 s, the built-in preconditioned GMRES 13 s and PETSc
   with algebraic multigrid on the momentum block (``gamg``) 5 s, in one
   thread.  ``linear_solver="automatic"`` therefore uses the built-in version
   for a Taylor-Hood system too large for an inexpensive direct factorisation
   (more than 100 000 unknowns in two dimensions, 4000 in three), and reports
   this choice when ``verbose=True``.  The preconditioner is available for the
   serial solver and the methods ``fem``, ``dmcdm`` and ``hfvm``.

For use with an external solver, :meth:`~dualmesh.Problem.linear_system` returns
the residual and the Jacobian of the current Newton step as a NumPy array and a
SciPy sparse matrix.

Reading the result
~~~~~~~~~~~~~~~~~~

A :class:`~dualmesh.SolveResult` carries:

``converged``
   Whether the iteration met one of the tolerances.

``total_iterations``
   Nonlinear iterations taken, summed over load steps and time steps.

``linear_iterations``
   Krylov iterations taken, summed the same way.  Zero for the direct solver.

``history``
   The residual norm at every nonlinear iteration.  It is the first diagnostic
   to examine when a solve fails.

``time_steps``, ``rejected_steps``, ``step_history``
   Filled by a transient solve (see below).

The transient solve
-------------------

.. code-block:: python

   result = problem.solve_transient(end_time=100.0, dt=1.0, theta=0.5)

Time is discretised with the :math:`\theta` method.  Writing :math:`M` for the
storage term and :math:`R_{\text{steady}}` for everything else, one step solves

.. math::

   M \frac{U^{n+1} - U^{n}}{\Delta t}
   + \theta R_{\text{steady}}(U^{n+1}, t^{n+1})
   + (1 - \theta) R_{\text{steady}}(U^{n}, t^{n}) = 0 .

``theta`` selects the member of the family.

:math:`\theta = 1`
   Backward Euler, the default.  First-order accurate in time and
   unconditionally stable, and strongly damping, which makes it robust on a
   stiff problem and on the first step after a discontinuous start.

:math:`\theta = 1/2`
   Crank-Nicolson [CrankNicolson1947]_.  Second-order accurate and
   unconditionally stable, but only neutrally damping, so a sharp initial
   transient can produce oscillations that persist for several steps.

:math:`\theta = 0`
   Forward Euler.  The steady terms are evaluated entirely at the old state, so
   the scheme is explicit and only conditionally stable.  For the slab problem
   of the test suite the critical step is between :math:`0.2 h^2` and
   :math:`0.3 h^2`, and above it the solution diverges within a few steps.

``start_time`` (default zero) sets the initial time, and ``output_interval``
with ``output_file_base`` writes a VTU file every ``output_interval`` accepted
steps.  All
the steady options above are accepted too and apply to the nonlinear solve
inside each step.

Choosing the time step
~~~~~~~~~~~~~~~~~~~~~~

``time_stepper`` selects how :math:`\Delta t` is chosen after the first step.

``"fixed"``
   The default.  The step stays at ``dt``, except that the last step is
   shortened so that the run ends exactly at ``end_time``.

``"error"``
   The step is chosen so that an estimate of the *local truncation error* stays
   below ``error_tolerance`` (default :math:`10^{-3}`).  The estimate comes from
   step doubling: the step is taken once with :math:`\Delta t` and again as two
   steps of :math:`\Delta t / 2`, and the difference between the two answers,
   scaled by the order of the method, estimates the error of the coarser one
   [RichardsonGaunt1927]_.  The next step is then set by the standard controller
   of [HairerNorsettWanner1993]_.  It costs three solves per accepted step, and
   it is the only stepper that measures the accuracy, so it is the stepper to
   use when the solution must meet a stated tolerance.  Tightening
   ``error_tolerance`` shortens the steps and reduces the error until the
   spatial discretisation error dominates, beyond which further tightening only
   increases the run time.

``"iteration"``
   The step is chosen from the number of nonlinear iterations that the last
   step needed, in the manner of MOOSE's ``IterationAdaptiveDT``
   [MOOSE2025]_.  A step that converged in fewer than
   ``optimal_iterations - iteration_window`` nonlinear iterations is followed
   by a longer one, and a step that needed more than
   ``optimal_iterations + iteration_window`` is followed by a shorter one.  It
   adds no cost beyond the solve itself.  It controls the computational
   *effort* and leaves the error uncontrolled, so it suits a nonlinear problem
   whose difficulty varies through the run and is unsuitable when accuracy is
   the concern.

``growth_factor`` (default 2.0) limits the growth of the step in one update,
and ``dt_min`` and ``dt_max`` set its lower and upper bounds.  ``dt_max`` at
its default of zero means no upper bound.

Rejecting and retrying a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A step is rejected when its nonlinear solve fails to converge, or, with the
error stepper, when its estimated error exceeds the tolerance.  A rejected step
is discarded entirely (the state is restored to the beginning of the step) and
retried with the step multiplied by ``cutback_factor`` (default 0.5).

``max_rejected_steps`` (default 10) limits how many consecutive rejections are
allowed before the run stops, and ``dt_min`` sets the other limit: a step that
has been cut back below it cannot be cut back again.  In either case the run
ends with an error stating that the transient solve failed, so that the step is
never reduced indefinitely.  ``result.rejected_steps`` counts the rejections,
and ``result.step_history`` lists every accepted ``(time, dt)`` pair, whose plot
shows the behaviour of the step controller.

A worked example
~~~~~~~~~~~~~~~~

The example is a steel slab, initially at 300 K, with one face held at 3000 K
and the other radiating to an ambient of 300 K.  The radiation makes the problem
nonlinear and the first step difficult to converge, so the iteration stepper is
deliberately given an initial step that is far too large, and it reduces the
step itself.

.. code-block:: python

   import dualmesh as dm
   import numpy as np

   mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
   problem = dm.Problem(mesh)
   problem.add_variable("temperature")
   problem.set_values("temperature", np.full(mesh.num_nodes, 300.0))

   problem.add_kernel("heat_conduction", "conduction",
                      variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel("heat_conduction_time_derivative", "storage",
                      variable="temperature", density=7800.0, specific_heat=460.0)
   problem.add_boundary_condition("Dirichlet_boundary_condition", "hot", variable="temperature",
                                  boundary="left", value=3000.0)
   problem.add_boundary_condition("radiative_heat_flux_boundary_condition", "radiation",
                                  variable="temperature", boundary="right",
                                  emissivity=0.9, ambient_temperature=300.0)

   result = problem.solve_transient(
       end_time=400.0, dt=400.0, theta=1.0,
       time_stepper="iteration", max_iterations=2, cutback_factor=0.25)

   print(result.converged, result.rejected_steps, len(result.step_history))

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
   for which the remedies are load stepping, a better initial condition, or
   ``nonlinear_solver="picard"`` with under-relaxation.

*"the conjugate gradient method ... requires a symmetric positive definite matrix"*
   ``linear_solver="cg"`` was used with a method that does not produce one.  Use
   the default, ``"automatic"``, or ``"bicgstab"``, ``"gmres"`` or ``"lu"``.

*"transient solve failed"*
   The step was cut back to ``dt_min`` or rejected ``max_rejected_steps`` times
   in a row and still did not converge.  Some problems have no bounded solution,
   in which case no step size leads to convergence, so check the physics before
   lowering ``dt_min`` any further.

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
which is one whenever any part of the problem is defined in Python.  Distributed
runs use :class:`~dualmesh.DistributedProblem`, and both forms of parallelism
are described in :doc:`/theory/parallel`.
