Solving
=======

Once the variables, kernels, boundary conditions and materials are in place, the
problem is solved either once, for a steady state, or repeatedly in time.  This
chapter explains both, and every option each takes.

.. contents::
   :local:
   :depth: 2

The steady solve
----------------

.. code-block:: python

   result = problem.solve()
   print(result.converged, result.total_iterations)

:meth:`~dualmesh.Problem.solve` assembles the residual and the Jacobian,
solves the linear system, and repeats until the residual is small enough.  It
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
   [Wengert1964]_ rather than differentiated by hand or approximated by finite
   differences, so the iteration converges quadratically: the number of correct
   digits doubles at each step once the iterate is close enough.  Watch
   ``result.history`` and you will see it.

``"picard"``
   Direct iteration, also called successive substitution.  The nonlinear
   coefficients are evaluated at the previous iterate, the resulting linear
   problem is solved, and the answer becomes the next iterate.  It converges
   linearly rather than quadratically, but each iteration is cheaper and its
   basin of attraction is often wider, so it is the method to try when Newton
   diverges from a poor starting point.  It is what the ``relaxation`` option
   below applies to.

``"linear"``
   Assemble once and solve once, with no iteration.  Use it when the problem is
   known to be linear; it saves the one extra residual evaluation that Newton
   spends confirming convergence.  If the problem is not in fact linear, the
   answer is simply the first Newton step and will be wrong, so this is an
   assertion about the problem rather than a shortcut.

Convergence and its tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The iteration stops when any of three tests passes.

``relative_tolerance`` (default :math:`10^{-10}`)
   The norm of the residual has fallen by this factor from its value at the
   first iteration.  This is the usual test, and it is scale free.

``absolute_tolerance`` (default :math:`10^{-12}`)
   The norm of the residual is below this outright.  This catches the case where
   the initial residual is already tiny, where the relative test would otherwise
   demand impossible precision.

``step_tolerance`` (default :math:`10^{-12}`)
   The norm of the correction is below this.  A problem whose residual cannot be
   driven down — because of a badly scaled equation, say — may still have
   stopped moving, and this test recognises that.

``max_iterations`` (default 50) caps the count.  ``error_on_divergence``
(default true) decides what happens when the cap is reached: with it true an
exception is raised, and with it false the result is returned with
``converged`` false so that the caller can decide.  Set it false inside a loop
that is expected to fail sometimes, such as a continuation study, and leave it
true otherwise.

``relaxation`` applies to direct iteration only and defaults to zero, meaning no
relaxation.  With :math:`\gamma` set, the coefficients are evaluated not at the
last iterate but at

.. math::

   \bar{U} = (1 - \gamma) U^{(r)} + \gamma U^{(r-1)} ,

so a value between zero and one under-relaxes and damps an oscillating
iteration, while a value above one over-relaxes and accelerates a slow
monotone one.  Under-relaxation is the usual remedy for a direct iteration that
oscillates without settling.

``verbose`` prints the residual norm at every iteration, which is the quickest
way to tell a slow convergence from a stalled one.

Load stepping
~~~~~~~~~~~~~

A nonlinear problem that will not converge from a zero initial guess will often
converge if the load is applied in pieces, each solve starting from the answer
to the last.

.. code-block:: python

   result = problem.solve(load_factors=[0.25, 0.5, 0.75, 1.0])

Each factor multiplies every object that was added with
``scale_with_load=True``, so the terms that represent applied load are ramped
and the terms that represent stiffness are not.  The factors need not be evenly
spaced, and the last one should normally be 1.0.  This is how the von Kármán
plate and beam problems of the solid mechanics module are driven to large deflection.

.. _linear-solver-choice:

The linear solver
~~~~~~~~~~~~~~~~~

``linear_solver`` chooses how each linearised system :math:`J \, \delta U =
-R` is solved.  The default, ``"automatic"``, is the right choice for almost
every problem; the others are there for control and for comparison.

``"automatic"``
   The default.  It factorises the system directly where that is cheap and
   iterates where it is not, and it falls back to the direct solver if the
   iteration fails, so it is never less robust than ``"lu"``.

   Where the line falls follows from how a sparse factorisation fills in.
   With a good ordering, the factors of a two-dimensional mesh problem with
   :math:`n` unknowns hold :math:`O(n \log n)` entries and cost
   :math:`O(n^{3/2})` operations, but those of a three-dimensional problem hold
   :math:`O(n^{4/3})` entries and cost :math:`O(n^2)` operations
   [George1973]_ [LiptonRoseTarjan1979]_.  A preconditioned Krylov iteration
   costs a few matrix-vector products, each :math:`O(n)`, per iteration.  So
   ``"automatic"`` factorises every one-dimensional system, two-dimensional
   systems up to :math:`10^5` unknowns, and three-dimensional systems up to a
   few thousand, and otherwise runs BiCGSTAB with an ILU(0) preconditioner.

   The difference is large in three dimensions.  For the three-dimensional
   elasticity problem of the verification suite on a :math:`12^3`-cell
   tetrahedral mesh, the cell-centred method has 36 288 unknowns and 1.3
   million matrix entries.  The direct solver takes 18 s (SuperLU, through
   SciPy, with the same column ordering, stores 47 times as many entries in
   the factors as in the matrix); BiCGSTAB with ILU(0) reaches a relative
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
   The Galerkin finite element method produces one; the dual mesh control domain
   method and both finite volume methods do not, because a control volume
   balance is not a weighted residual and the coupling from node :math:`I` to
   node :math:`J` need not match the coupling from :math:`J` to :math:`I`.
   Selecting ``"cg"`` with one of those methods produces a failure to converge,
   and the error message says exactly this rather than leaving you to guess.

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
   Division by the diagonal, and no preconditioning.  For comparison.

To use a solver of your own, :meth:`~dualmesh.Problem.linear_system` returns
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
   The residual norm at every nonlinear iteration.  This is the diagnostic to
   look at first when something goes wrong.

``time_steps``, ``rejected_steps``, ``step_history``
   Filled by a transient solve; see below.

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
   unconditionally stable, and strongly damping, which makes it forgiving on a
   stiff problem and on the first step after a discontinuous start.

:math:`\theta = 1/2`
   Crank-Nicolson [CrankNicolson1947]_.  Second-order accurate and
   unconditionally stable, but only neutrally damping, so a sharp initial
   transient can ring for several steps before it settles.

:math:`\theta = 0`
   Forward Euler.  The steady terms are evaluated entirely at the old state, so
   the scheme is explicit and only conditionally stable.  For the slab problem
   of the test suite the critical step is between :math:`0.2 h^2` and
   :math:`0.3 h^2`; above that the solution diverges within a few steps.

``start_time`` (default zero) sets the initial time, and ``output_interval``
with ``output_file_base`` writes a VTU file every that-many accepted steps.  All
the steady options above are accepted too and apply to the nonlinear solve
inside each step.

Choosing the time step
~~~~~~~~~~~~~~~~~~~~~~

``time_stepper`` selects how :math:`\Delta t` is chosen after the first step.

``"fixed"``
   The default.  The step stays at ``dt``, except that the last step is
   shortened so that the run lands exactly on ``end_time``.

``"error"``
   The step is chosen so that an estimate of the *local truncation error* stays
   below ``error_tolerance`` (default :math:`10^{-3}`).  The estimate comes from
   step doubling: the step is taken once with :math:`\Delta t` and again as two
   steps of :math:`\Delta t / 2`, and the difference between the two answers,
   scaled by the order of the method, estimates the error of the coarser one
   [RichardsonGaunt1927]_.  The next step is then set by the standard controller
   of [HairerNorsettWanner1993]_.  It costs three solves per accepted step and
   it is the only stepper that measures accuracy rather than guessing at it, so
   it is what to use when the answer has to be right to a stated tolerance.
   Tightening ``error_tolerance`` shortens the steps and reduces the error until
   the spatial discretisation error takes over, at which point tightening it
   further only costs time.

``"iteration"``
   The step is chosen from how hard the last step was to solve, in the manner of
   MOOSE's ``IterationAdaptiveDT`` [MOOSE2025]_.  A step that converged in fewer
   than ``optimal_iterations - iteration_window`` nonlinear iterations is
   followed by a longer one; a step that needed more than
   ``optimal_iterations + iteration_window`` is followed by a shorter one.  It
   costs nothing beyond the solve itself, and it controls *effort* rather than
   error, so it is the right stepper for a nonlinear problem whose difficulty
   varies through the run and the wrong one when accuracy is the concern.

``growth_factor`` (default 2.0) caps how much the step may grow in one go, and
``dt_min`` and ``dt_max`` bound it outright.  ``dt_max`` at its default of zero
means no upper bound.

Rejecting and retrying a step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A step is rejected when its nonlinear solve fails to converge, or, with the
error stepper, when its estimated error exceeds the tolerance.  A rejected step
is thrown away entirely — the state is restored to the beginning of the step —
and retried with the step multiplied by ``cutback_factor`` (default 0.5).

``max_rejected_steps`` (default 10) limits how many times in a row this may
happen before the run gives up, and ``dt_min`` is the other stop: a step that
has been cut back below it cannot be cut back again.  Either way the run ends
with an error that says the transient solve failed, rather than halving the step
forever.  ``result.rejected_steps`` counts the rejections and
``result.step_history`` lists every accepted ``(time, dt)`` pair, which is what
to plot when you want to see what the controller did.

A worked example
~~~~~~~~~~~~~~~~

A steel slab, initially at 300 K, with one face held at 3000 K and the other
radiating to an ambient of 300 K.  The radiation makes the problem nonlinear and
the first step very hard, so the iteration stepper is given a step far too large
on purpose and left to find its own way down.

.. code-block:: python

   import dualmesh as dm
   import numpy as np

   mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
   problem = dm.Problem(mesh)
   problem.add_variable("temperature")
   problem.set_values("temperature", np.full(mesh.num_nodes, 300.0))

   problem.add_kernel("HeatConduction", "conduction",
                      variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel("HeatConductionTimeDerivative", "storage",
                      variable="temperature", density=7800.0, specific_heat=460.0)
   problem.add_boundary_condition("DirichletBC", "hot", variable="temperature",
                                  boundary="left", value=3000.0)
   problem.add_boundary_condition("RadiativeHeatFluxBC", "radiation",
                                  variable="temperature", boundary="right",
                                  emissivity=0.9, ambient_temperature=300.0)

   result = problem.solve_transient(
       end_time=400.0, dt=400.0, theta=1.0,
       time_stepper="iteration", max_iterations=2, cutback_factor=0.25)

   print(result.converged, result.rejected_steps, len(result.step_history))

When a solve fails
------------------

The error messages are written to say what to do, not only what happened.  The
ones worth knowing in advance:

*"Newton did not converge"*
   The residual stopped falling.  Look at ``result.history``: a residual that
   falls and then flattens usually means an inconsistent Jacobian, which in this
   library means a kernel that computed something in plain ``float`` instead of
   :class:`~dualmesh.ADReal`.  A residual that grows means the iterate has left
   the basin of attraction, for which the remedies are load stepping, a better
   initial condition, or ``nonlinear_solver="picard"`` with under-relaxation.

*"the conjugate gradient method ... requires a symmetric positive definite matrix"*
   ``linear_solver="cg"`` was used with a method that does not produce one.  Use
   the default, ``"automatic"``, or ``"bicgstab"``, ``"gmres"`` or ``"lu"``.

*"transient solve failed"*
   The step was cut back to ``dt_min`` or rejected ``max_rejected_steps`` times
   in a row and still would not converge.  Some problems have no bounded
   solution and no step size helps; check the physics before lowering ``dt_min``
   any further.

*"degenerate element (zero Jacobian determinant)"*
   An element has zero or inverted volume.  Call ``mesh.fix_orientation()``,
   which repairs a consistently inverted mesh, and check any
   ``transform_nodes`` that may have folded the mesh over itself.

*"Too many variables for this mesh"*
   The automatic differentiation budget is exhausted; the message gives the
   arithmetic and the CMake setting that raises it.

Running in parallel
-------------------

Both solves thread their assembly loops automatically.
:meth:`~dualmesh.Problem.set_num_threads` sets the count and
:meth:`~dualmesh.Problem.effective_threads` reports what will actually be used,
which is one whenever any part of the problem is defined in Python.  Distributed
runs use :class:`~dualmesh.DistributedProblem` instead; both are described in
:doc:`/theory/parallel`.
