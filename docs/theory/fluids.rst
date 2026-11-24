Incompressible flow
===================

This page states the equations that the ``fluid_dynamics`` physics solves for
the flow of an incompressible Newtonian fluid, and how the code discretizes
them.  It covers the three formulations of the pressure, the stabilization of
equal-order elements, the advection of momentum, the two ways of advancing a
transient flow, and the buoyancy that couples the flow to the temperature.
Both methods of the code assemble these equations: the finite element method
(FEM) and the dual mesh control domain method (DMCDM) [Reddy2024]_.  A flow
takes DMCDM when the method is not given.

The governing equations
-----------------------

The flow conserves mass and momentum.  With :math:`\mathbf{v}` the velocity,
:math:`p` the pressure, :math:`\rho` the constant density, :math:`\mu` the
dynamic viscosity and :math:`\mathbf{f}` the body force per unit volume in
:math:`\mathrm{N/m^3}`, the two balances are

.. math::
   :label: flow-mass

   \nabla \cdot \mathbf{v} = 0

and

.. math::
   :label: flow-momentum

   \rho \left( \frac{\partial \mathbf{v}}{\partial t}
   + (\mathbf{v} \cdot \nabla) \mathbf{v} \right)
   - \nabla \cdot \boldsymbol{\sigma} = \mathbf{f} ,
   \qquad
   \boldsymbol{\sigma} = -p \mathbf{I}
   + \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}} \right) ,

with :math:`\boldsymbol{\sigma}` the Cauchy stress and :math:`\mathbf{I}` the
identity.  Equation :eq:`flow-mass` states that the volume of a material
element does not change.  The first term of :eq:`flow-momentum` is the mass
per unit volume times the acceleration, the rate of change at a point plus the
convective term :math:`(\mathbf{v} \cdot \nabla) \mathbf{v}`.  With
:eq:`flow-mass`, the momentum equation of the component :math:`v_i` is the
balance law :math:`\partial m / \partial t + \nabla \cdot \mathbf{F} = s` with
the storage :math:`m = \rho v_i`, the flux
:math:`\mathbf{F} = \rho v_i \mathbf{v} - \boldsymbol{\sigma} \mathbf{e}_i` and
the source :math:`s = f_i`, where :math:`\mathbf{e}_i` is the unit vector of
the direction :math:`i`.  For a flow of speed :math:`V` across a length
:math:`L`, the Reynolds number :math:`Re = \rho V L / \mu` compares the
convective term with the viscous term.

The viscous term has two forms, chosen by ``viscous_form``:

* ``"stress"`` (the default) takes the stress above.  Its natural boundary
  quantity is the traction :math:`\boldsymbol{\sigma} \mathbf{n}`, so a
  boundary without a condition is traction free.
* ``"laplacian"`` replaces the viscous part
  :math:`\mu (\nabla v_i + \partial \mathbf{v} / \partial x_i)` of
  :math:`\boldsymbol{\sigma} \mathbf{e}_i` by :math:`\mu \nabla v_i`.  The two give
  the same equations for a constant viscosity and a divergence-free velocity.
  A boundary without a condition then has
  :math:`\mu\, \partial \mathbf{v} / \partial n - p \mathbf{n} = \mathbf{0}`,
  the do-nothing outflow condition.

In axisymmetric coordinates :math:`(r, z)` without swirl, the radial equation
also holds the hoop stress :math:`\sigma_{\theta\theta} = -p + 2 \mu v_r / r`,
with :math:`v_r` the radial velocity: its source is
:math:`s = f_r - \sigma_{\theta\theta} / r`.  The ``"laplacian"`` form takes
:math:`\mu v_r / r^2` in place of :math:`2 \mu v_r / r^2`, the term of the
radial component of the vector Laplacian.

The formulations
----------------

The pressure has no equation of its own.  It is the Lagrange multiplier of the
constraint :eq:`flow-mass`: it takes the value that keeps the flow divergence
free, and an enclosed flow fixes it only up to a constant.  A discretization
that keeps the pressure as an unknown must satisfy the inf-sup, or
Ladyzhenskaya-Babuška-Brezzi, condition [BoffiBrezziFortin2013]_: every discrete
pressure other than a constant must do work on some discrete velocity through
the divergence.  Interpolating the velocity and the pressure with the same
functions violates it.  On a mesh of quadrilaterals the checkerboard pressure,
which alternates between :math:`+1` and :math:`-1` from node to node, exerts
no force on any discrete velocity, so the discrete equations do not determine
it.

The parameter ``formulation`` of ``fluid_dynamics`` chooses one of three
remedies:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Formulation
   * - ``"equal_order"``
     - velocity and pressure of the order of the mesh, with a stabilization of
       the mass equation.  The default, except as below.
   * - ``"Taylor-Hood"``
     - quadratic velocity and linear pressure, stable without stabilization.
       It needs ``method="FEM"`` and a quadratic mesh, and it is the default
       when you give ``method="FEM"`` on a quadratic mesh with the monolithic
       time integration.
   * - ``"penalty"``
     - the pressure eliminated by :math:`p = -\gamma \nabla \cdot \mathbf{v}`,
       which makes the flow slightly compressible.

The input takes ``equal_order`` (its default) or ``Taylor-Hood`` as the
``formulation`` of ``fluid_dynamics``, with the properties ``density`` and
``dynamic_viscosity``.  A ``density`` of 0, the default of the Python
physics, gives a Stokes flow: the physics then adds no inertia.

The equal-order formulation
---------------------------

The kernels
~~~~~~~~~~~

For every velocity component, ``fluid_dynamics`` adds ``viscous_stress`` (the
viscous flux), ``pressure_gradient`` (the pressure part :math:`p \mathbf{e}_i`
of the flux, with the hoop term in axisymmetric coordinates), and, with a
positive density, ``convective_inertia``, ``momentum_stabilization`` and
``time_derivative`` (the term :math:`\rho\, \partial v_i / \partial t` of a
transient solve).  A ``body_force`` adds :math:`f_i`.  The mass equation is the
kernel ``mass_conservation`` on the pressure variable, and the boundary
condition ``mass_flux`` on every side set closes it.

The stabilized mass flux
~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\mathbf{r}_m`, in :math:`\mathrm{N/m^3}`, be the residual of the
momentum equation in its strong form,

.. math::
   :label: flow-residual

   \mathbf{r}_m = \rho \left( \frac{\partial \mathbf{v}}{\partial t}
   + (\mathbf{v} \cdot \nabla) \mathbf{v} \right) + \nabla p
   - \mu \nabla^2 \mathbf{v} - \mathbf{f} .

``mass_conservation`` writes the mass equation as the balance law with no
storage, no source and the flux

.. math::
   :label: flow-mass-flux

   \mathbf{F} = \mathbf{v} - \tau\, \mathbf{r}_m .

The added term :math:`-\tau \mathbf{r}_m` vanishes for the exact solution.
For FEM, testing :math:`\nabla \cdot \mathbf{F} = 0` with a pressure shape
function :math:`\psi` and integrating by parts adds
:math:`\int_\Omega \tau\, \nabla \psi \cdot \mathbf{r}_m \,\mathrm{d}V` to the
Galerkin mass equation: the pressure-stabilizing Petrov-Galerkin (PSPG) method
of [HughesFrancaBalestra1986]_.  Its part
:math:`\int_\Omega \tau\, \nabla \psi \cdot \nabla p \,\mathrm{d}V` is a small
pressure Laplacian that removes the checkerboard mode.  For DMCDM, the same
flux crosses the faces of each control domain, so the face mass flux is the
interpolated velocity corrected by :math:`\tau \mathbf{r}_m`.  This extends the
momentum interpolation of [RhieChow1983]_ from the pressure gradient to the
whole momentum residual.

The stabilization parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter :math:`\tau`, in :math:`\mathrm{s\,m^3/kg}`, is that of
[Tezduyar1991]_ (see also [TezduyarMittalRayShih1992]_), written per unit
density so that it also covers a Stokes flow:

.. math::
   :label: flow-tau

   \tau = \left[ \left( \frac{2 \rho}{\Delta t} \right)^2
   + \left( \frac{2 \rho |\mathbf{v}|}{h} \right)^2
   + 9 \left( \frac{4 \mu}{h^2} \right)^2 \right]^{-1/2} .

The three terms are the reciprocal time scales of the unsteadiness, of the
convection across an element and of the diffusion across it, each times
:math:`\rho`.  The first term enters a transient solve only.  In the steady
viscous limit :math:`\rho \tau = h^2 / (12 \mu / \rho)`, the limit of the optimal
parameter :math:`(h / 2 |\mathbf{v}|)(\coth Pe_h - 1 / Pe_h)` of linear
elements [BrooksHughes1982]_, with the cell Péclet number
:math:`Pe_h = \rho |\mathbf{v}| h / (2 \mu)`.  The factor 9 gives this limit.
The speed :math:`|\mathbf{v}|` is that of the current iterate under Newton's
method and of the previous one under Picard iteration.  The element size is
:math:`h = (c V_e)^{1/d}`, with :math:`V_e` the volume of the element,
:math:`d` the dimension of the mesh and :math:`c = 1` for quadrilaterals and
hexahedra, 2 for triangles and prisms, and 6 for tetrahedra and pyramids, so
that each element of a structured grid of spacing :math:`h` has the size
:math:`h`.  A quadratic element has half that size.

Consistency
~~~~~~~~~~~

The viscous term of :eq:`flow-residual` needs second derivatives.  Quadratic
elements give the Laplacian of their shape functions, on curved elements too,
with the terms of the axisymmetric and spherical Laplacians.  For a viscosity
uniform in space, the residual of the exact solution is then zero, and the
stabilization is consistent.  For a viscosity that varies in space the term
lacks :math:`\nabla \mu \cdot (\nabla \mathbf{v} + \nabla
\mathbf{v}^{\mathsf{T}})`.  Linear elements leave the viscous term out,
because the Laplacian of their interpolant vanishes inside them.  The residual
of the exact solution on linear elements is therefore
:math:`-\mu \nabla^2 \mathbf{v}`, and the stabilization adds the term
:math:`\tau \mu \nabla^2 \mathbf{v}` to the mass flux.  In the steady viscous
limit this term is :math:`h^2 \nabla^2 \mathbf{v} / 12`, of order :math:`h^2`.

The streamline stabilization of the momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With inertia, ``momentum_stabilization`` adds the streamline upwind
Petrov-Galerkin (SUPG) term of [BrooksHughes1982]_ to the equation of each
component :math:`v_i`, as the flux :math:`-\rho \tau\, r_{m,i}\, \mathbf{v}`.
For FEM it adds
:math:`\int_\Omega \rho \tau\, (\mathbf{v} \cdot \nabla \psi)\, r_{m,i}
\,\mathrm{d}V` to the equation of the test function :math:`\psi`, and for DMCDM
it is a streamline diffusion across the faces of the control domains.  It
damps the oscillations of a flow that convection dominates on a coarse mesh.
``fluid_dynamics`` adds it to an equal-order flow whenever the density is
positive.

Every force of the momentum equations must also enter :math:`\mathbf{r}_m`,
because a force missing there makes the stabilization act against it.
``mass_conservation`` and ``momentum_stabilization`` read the density, the
viscosity, the body forces and the Boussinesq buoyancy from the kernels
``convective_inertia``, ``time_derivative``, ``viscous_stress``,
``body_force`` and ``Boussinesq_buoyancy`` of the velocities.  They refuse
any kernel on a velocity outside these, ``pressure_gradient``,
``momentum_stabilization`` and ``penalty_incompressibility``, because
:math:`\mathbf{r}_m` would miss its term.

.. _conserved-advection:

The conserved velocity
~~~~~~~~~~~~~~~~~~~~~~

The flux :eq:`flow-mass-flux` of the stabilized mass equation is the velocity

.. math::
   :label: flow-conserved-velocity

   \mathbf{w} = \mathbf{v} - \tau\, \mathbf{r}_m ,

the coarse velocity plus the fine-scale velocity :math:`-\tau \mathbf{r}_m` of
[HughesWells2005]_.  The discrete flow conserves the flux of
:math:`\mathbf{w}`, and of :math:`\mathbf{v}` only up to
:math:`\tau \mathbf{r}_m`.  ``convective_inertia`` and ``heat_convection``
therefore advect with :math:`\mathbf{w}` in an equal-order flow, in the
advective form.  FEM integrates
:math:`\rho N_j\, (\mathbf{w} \cdot \nabla) v_i` for the equation of node
:math:`j`.  DMCDM sums
:math:`\rho (\mathbf{w} \cdot \mathbf{A}_f)(v_{i,f} - v_{i,j})` over the faces
:math:`f` of the control domain :math:`V_j` of node :math:`j`, with
:math:`\mathbf{A}_f` the outward area vector of the face and :math:`v_{i,f}`
the face value of the advection scheme (:doc:`finite_volume`).  On the boundary
of the domain DMCDM takes :math:`\mathbf{v}`, as the mass equation does.

The mass equation holds for every test function of the velocity, because the
pressure is interpolated like it.  Two properties follow:

* The momentum and the heat are conserved exactly against the discrete mass
  flux: summed over all nodes, the advective term is what the mass flux carries
  out of the domain, through the boundary and through the nodes whose pressure
  is prescribed.
* A uniform velocity is carried unchanged, and the advected temperature does
  not depend on its scale, because the term has no part
  :math:`T \nabla \cdot \mathbf{w}`.

Under Picard iteration the velocity part of :math:`\mathbf{w}` is that of the
previous iterate.  In a step of the theta method, the time derivative in
:math:`\mathbf{w}` has the weight 1 at the new time.  At the old time of a
step, :math:`\mathbf{w}` is the velocity that the mass equation of the previous
step conserved, and the first step carries :math:`\mathbf{v}` there.  The
penalty and Taylor-Hood formulations have no equal-order mass equation, so they
advect with :math:`\mathbf{v}`, which conserves momentum and heat only as far
as the discrete velocity is divergence free.

The advection schemes of the momentum
-------------------------------------

The ``fluid_dynamics`` physics assembles the inertia with the central value of
the velocity, and an equal-order flow adds the streamline stabilization above,
with no option.  The
object ``convective_inertia`` takes the parameter ``advection_scheme`` for a
flow assembled object by object:

* ``"central"`` (the default): the value at the point, for both methods.
* ``"upwind"``, ``"linear_upwind"`` and ``"limited"``: the face fluxes of
  DMCDM of :doc:`finite_volume`, with the upstream value (first order), the
  upstream value corrected by the upstream gradient (second order), or the
  upstream value blended towards the central one by the van Albada limiter
  (second order and bounded).

The Taylor-Hood element
-----------------------

The element of [TaylorHood1973]_ makes the velocity space richer than the
pressure space.  It interpolates the velocity with the quadratic shape
functions of the element and the pressure with the linear shape functions of
its corners: :math:`P_2`-:math:`P_1` on triangles and tetrahedra and
:math:`Q_2`-:math:`Q_1` on quadrilaterals and hexahedra.  With :math:`V_h` and
:math:`Q_h` the discrete velocity and pressure spaces, the pair satisfies the
discrete inf-sup condition

.. math::
   :label: flow-inf-sup

   \inf_{\psi_h \in Q_h} \sup_{\mathbf{v}_h \in V_h}
   \frac{(\psi_h, \nabla \cdot \mathbf{v}_h)}
        {|\mathbf{v}_h|_1\, \|\psi_h\|_0} \ge c_{\mathrm{is}} > 0 ,

with the inf-sup constant :math:`c_{\mathrm{is}}` independent of :math:`h`
[BoffiBrezziFortin2013]_.  Here
:math:`(\cdot, \cdot)` is the :math:`L^2` inner product over :math:`\Omega`,
:math:`\|\cdot\|_0` the :math:`L^2` norm and :math:`|\cdot|_1` the
:math:`H^1` seminorm, the :math:`L^2` norm of the gradient.  The Galerkin
equations therefore determine a unique pressure with no stabilization.  On a
smooth solution the errors fall as [BoffiBrezziFortin2013]_

.. math::
   :label: flow-taylor-hood-rates

   \|\mathbf{v} - \mathbf{v}_h\|_0 = O(h^3) , \qquad
   |\mathbf{v} - \mathbf{v}_h|_1 = O(h^2) , \qquad
   \|p - p_h\|_0 = O(h^2) .

The mass equation takes the flux :math:`\mathbf{F} = \mathbf{v}`, tested with
the linear pressure shape functions, and the momentum equations are those of
the equal-order formulation without ``momentum_stabilization``, tested with
the quadratic shape functions.

The pressure is a variable declared with ``order="first"``.  On every quadratic
element it takes the shape functions :math:`N^c_a` of the linear element with
the same corners: ``Tri3`` for ``Tri6``, ``Quad4`` for ``Quad9`` and ``Quad8``,
``Tet4`` for ``Tet10``, and ``Hex8`` for ``Hex27`` and ``Hex20``.  Their
gradients use the Jacobian :math:`\mathbf{J}` of the quadratic map of the
element, :math:`\nabla N^c_a = \mathbf{J}^{-\mathsf{T}} \hat{\nabla} N^c_a`,
with :math:`\hat{\nabla}` the gradient in the reference coordinates.  On a
curved element the pressure is therefore linear in the reference coordinates.
The pressure values at the nodes that are not corners hold the linear field
there, set after every change of the solution, so that the quadratic shape
functions interpolate the linear pressure exactly in sampling and output.

The penalty formulation
-----------------------

The penalty formulation [HughesLiuBrooks1979]_ [Reddy2024]_ relaxes the
constraint :eq:`flow-mass` with the constitutive assumption

.. math::
   :label: flow-penalty

   p = -\gamma\, \nabla \cdot \mathbf{v} ,

with :math:`\gamma` the penalty parameter in :math:`\mathrm{Pa\,s}`.  A
compressed fluid element, with :math:`\nabla \cdot \mathbf{v} < 0`, has a
raised pressure.  As :math:`\gamma \to \infty`, a bounded pressure forces
:math:`\nabla \cdot \mathbf{v} \to 0`, and at a finite :math:`\gamma` the flow
keeps a dilatation of order :math:`p / \gamma`.  Substituting
:eq:`flow-penalty` into the stress removes the pressure from the unknowns: the
flux of the equation of :math:`v_i` gains
:math:`-\gamma (\nabla \cdot \mathbf{v})\, \mathbf{e}_i`, which
``penalty_incompressibility`` assembles.  In axisymmetric coordinates the
divergence holds :math:`v_r / r`, and the hoop stress holds the penalty
pressure.

The parameter ``penalty_parameter`` defaults to :math:`10^8`, which suits a
viscosity of order one.  Choose :math:`\gamma` between about :math:`10^4 \mu`
and :math:`10^7 \mu`, with :math:`\mu` in :math:`\mathrm{Pa\,s}`.  A smaller
:math:`\gamma` leaves a dilatation large enough to change the flow.  A larger
one makes the
matrix so ill-conditioned that even a direct factorization loses accuracy.

The penalty term needs a reduced quadrature rule.  A rule of :math:`n` points
per element imposes, as :math:`\gamma \to \infty`, the constraint
:math:`\nabla \cdot \mathbf{v} = 0` at each point.  A structured mesh of
:math:`N_e` four-node quadrilaterals has about :math:`2 N_e` velocity unknowns
in two dimensions.  The :math:`2 \times 2` Gauss rule imposes :math:`4 N_e`
constraints on them, which only the zero velocity satisfies: the flow locks,
and refining the mesh does not help, because it multiplies constraints and
unknowns in the same proportion.  The one-point rule imposes :math:`N_e`
constraints, two unknowns per constraint, the ratio of the continuum problem.
``penalty_incompressibility`` therefore defaults to the ``"midpoint"``
``quadrature`` and to ``reduced_integration = true``, which evaluates the
velocity gradients at the centroid of the element.  This is selective reduced
integration [MalkusHughes1978]_: the viscous term keeps its full rule.

The property object ``penalty_pressure``, which ``fluid_dynamics`` adds,
declares the property ``pressure`` as :math:`-\gamma \nabla \cdot \mathbf{v}`,
with the :math:`\gamma` of the ``penalty_incompressibility`` kernels of the
velocities.  It is most accurate at the element centroids, where the
constraint is imposed.  The input does not offer this formulation.

Boundary conditions and the pressure level
------------------------------------------

Walls, inlets and moving lids carry conditions on the velocity components.  A
boundary
without a velocity condition is traction free in the ``"stress"`` form, the
usual outlet.  The mass equation needs the flow through the boundary:
``mass_flux`` gives its natural boundary quantity :math:`\mathbf{v} \cdot
\mathbf{n}`, the volume flux leaving, so the stabilizing part of the mass flux
has no normal component there.  ``fluid_dynamics`` places it on every side
set.  Summed over all nodes, the mass equations give
:math:`\oint_{\partial\Omega} \mathbf{v}_h \cdot \mathbf{n} \,\mathrm{d}S = 0`,
with :math:`\mathbf{v}_h` the discrete velocity, so the discrete flow conserves
mass globally.

An enclosed flow, with a velocity condition on every boundary, determines the
pressure only up to a constant.  The solver detects it on the first Newton
iteration from the Jacobian :math:`\mathbf{J}` of the residual: the constant
pressure :math:`\mathbf{1}_p` solves the homogeneous problem, with
:math:`\|\mathbf{J} \mathbf{1}_p\| \le 10^{-9}\, \||\mathbf{J}|\,
\mathbf{1}_p\|` for every equation, where :math:`|\mathbf{J}|` holds the
absolute values of the entries.  The solver then holds the free pressure
unknown nearest the centre of all free pressure unknowns at its value.  After each converged solve, it
shifts the pressure to a mean of zero over the fluid and takes the reactions at
the shifted state.  When a viscosity, a density or a source depends on the
pressure, the level changes the flow, and the solve stops with an error that
asks for the pressure at a point,
``flow.add_boundary_condition("pressure = 0", point=[...])``, or for a
pressure or traction condition.

The held unknown replaces one mass equation, so a net flux
:math:`\oint_{\partial\Omega} \mathbf{v}_h \cdot \mathbf{n} \,\mathrm{d}S \ne 0`
of the prescribed velocities can only enter there.  The solve then warns with
the size of the net flux.  A flow with velocities that cross its boundary needs
an outlet: a boundary with a traction condition, or no condition, determines
the pressure.

The steady solve
----------------

The inertia :math:`\rho (\mathbf{v} \cdot \nabla) \mathbf{v}` makes the
equations nonlinear.  Newton's method takes the Jacobian from forward automatic
differentiation of the residual, so it contains both parts of the linearized
inertia, and it converges quadratically near the solution.  Picard iteration
takes the transporting velocity from the previous iterate, so each iteration
solves a linear problem.  A steady solve of a flow at rest, with zero velocity
at every unknown that no condition prescribes, starts Newton's method from the
Stokes solution, the same problem without the inertia.  :doc:`nonlinear_and_time`
describes the two iterations, their relaxation and the continuation that a
steady solve applies when Newton's method fails to converge.

Transient flow
--------------

A transient solve keeps the inertia :math:`\rho\, \partial \mathbf{v} /
\partial t`, which ``fluid_dynamics`` adds to every momentum equation when the
density is positive.  The parameter ``time_integration`` chooses how the flow
advances.

The monolithic time integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"monolithic"``, the default, applies the theta method of ``solve_transient``
(:doc:`nonlinear_and_time`) to the whole system: every step solves the velocity
and the pressure together by Newton's method, as a steady solve does.  The mass
equation has no time derivative, so every :math:`\theta` takes it at the new
time with the weight 1.  The pressure gradient and ``momentum_stabilization``
are also taken at the new time.  In a step, the residual :eq:`flow-residual`
is that of the discrete momentum equation: the time difference, the pressure
gradient at the new time, and the other terms weighted by :math:`\theta` at the
new state and by :math:`1 - \theta` at the old one.  It vanishes for the
solution of the discrete equations, so the stabilization is consistent in time.
The parameter :eq:`flow-tau` depends on :math:`\Delta t`, so a converged steady
flow is not a fixed point of a time step.

The projection time integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"projection"`` is the splitting of [Fischer2021]_.  It replaces the saddle
point system by symmetric positive definite problems for FEM, each with its
own fast solver.  Let :math:`\ell` be the ``time_order`` (1, 2 or 3, default
2), :math:`b_0, \dots, b_\ell` the dimensionless coefficients of the backward
difference formula of order :math:`\ell` (BDF\ :math:`\ell`), and
:math:`a_1, \dots, a_\ell` those of the extrapolation of order :math:`\ell`
(EXT\ :math:`\ell`) of the advection.  The superscript :math:`n` marks the
time level :math:`t^n` of the new step.  The step collects the known terms in

.. math::
   :label: flow-projection-star

   \mathbf{v}^* = -\sum_{j=1}^{\ell} \left[ b_j \mathbf{v}^{n-j}
   + \Delta t\, a_j (\mathbf{v}^{n-j} \cdot \nabla) \mathbf{v}^{n-j} \right]
   + \frac{\Delta t}{\rho} \mathbf{f}^n ,

and the momentum equation at :math:`t^n` reads

.. math::
   :label: flow-projection-momentum

   \frac{\rho\, b_0}{\Delta t} \mathbf{v}^n
   = \frac{\rho}{\Delta t} \mathbf{v}^* - \nabla p^n + \mu \nabla^2 \mathbf{v}^n .

The coefficients come from the time levels themselves:
:math:`b_j / \Delta t` is the derivative at :math:`t^n` of the Lagrange
polynomial of :math:`t^{n-j}` on the levels :math:`t^n, \dots, t^{n-\ell}`, and
:math:`a_j` the value at :math:`t^n` of the Lagrange polynomial of
:math:`t^{n-j}` on the levels :math:`t^{n-1}, \dots, t^{n-\ell}`.  A step
shortened to land on an output time therefore keeps the order.  The first steps
take the orders below :math:`\ell`.

**The pressure.**  Testing :eq:`flow-projection-momentum` with the gradient of
a test function :math:`\psi` gives an equation for the pressure.  For a
divergence-free velocity the viscous term is
:math:`\mu \nabla^2 \mathbf{v} = -\mu \nabla \times \boldsymbol{\omega}`, with
:math:`\boldsymbol{\omega} = \sum_j a_j \nabla \times \mathbf{v}^{n-j}` the
vorticity extrapolated from the earlier steps, in :math:`\mathrm{1/s}`.  The
identity :math:`\nabla \cdot (\boldsymbol{\omega} \times \nabla \psi) =
\nabla \psi \cdot \nabla \times \boldsymbol{\omega}` turns this term into an
integral over the boundary, and the pressure equation is

.. math::
   :label: flow-projection-pressure

   (\nabla \psi, \nabla p^n) = \frac{\rho}{\Delta t} (\nabla \psi, \mathbf{v}^*)
   - \frac{\rho\, b_0}{\Delta t} \int_{\partial\Omega} \psi\, \mathbf{n} \cdot
     \mathbf{v}_b \,\mathrm{d}S
   - \mu \int_{\partial\Omega} \mathbf{n} \cdot
     (\boldsymbol{\omega} \times \nabla \psi) \,\mathrm{d}S .

Here :math:`\mathbf{v}_b` is the velocity on the boundary: the velocity
extrapolated from the earlier steps, with the prescribed value at :math:`t^n`
where a condition prescribes it.  The first boundary term comes from
:math:`-(\rho b_0 / \Delta t)(\nabla \psi, \mathbf{v}^n)`, which is an integral
over the boundary for a divergence-free :math:`\mathbf{v}^n`.  The second is the curl-curl form of the viscous term, which
[Fischer2021]_ keeps to control the divergence error at the boundary.  It
needs only first derivatives of the velocity, so it applies to linear elements.
Where a condition prescribes the pressure, the equation takes that value.  An
enclosed flow has its pressure fixed at one node after the right-hand side is
made to sum to zero, and then shifted to a mean of zero.

**The velocity.**  Each component then solves the Helmholtz equation

.. math::
   :label: flow-projection-helmholtz

   \mu (\nabla \psi, \nabla v_i^n) + \frac{\rho\, b_0}{\Delta t} (\psi, v_i^n)
   = \frac{\rho}{\Delta t} (\psi, v_i^*)
   - \left( \psi, \frac{\partial p^n}{\partial x_i} \right) ,

with the prescribed velocity on its boundary and the natural condition
:math:`\mu\, \partial v_i / \partial n = 0` where the velocity is free.  The
viscous term is that of the ``"laplacian"`` form.  The step keeps the residual
of the momentum equations, so the reactions and the force of the fluid on a
wall are available as after a monolithic step.

**The two methods.**  FEM takes the shape functions as the test functions
:math:`\psi`.  DMCDM takes the indicator of the control domain :math:`V_i` of
each node, so that :math:`(\psi, g)` becomes
:math:`\int_{V_i} g \,\mathrm{d}V` and :math:`(\nabla \psi, \mathbf{G})`
becomes :math:`-\oint_{\partial V_i} \mathbf{G} \cdot \mathbf{n}
\,\mathrm{d}S` over the interfaces between control domains, for a scalar
:math:`g` and a vector :math:`\mathbf{G}`.  The mass and gradient terms are
then integrals over the control domains, and the Laplacian is the flux of the
gradient out of them.  The curl-curl term keeps the shape functions.  FEM
integrates with a Gauss rule of one point more per direction than the order of
the element.  DMCDM integrates with the midpoint rule of every part of a
control domain and of every interface on linear elements.

**Equal-order elements at small steps.**  Through
:math:`(\nabla \psi, \mathbf{v}^*)`, the divergence of the earlier velocities
enters the pressure equation with the weight :math:`\rho b_0 / \Delta t`, as
the pressure stabilization :eq:`flow-mass-flux` with
:math:`\tau = \Delta t / (\rho b_0)` would.  That parameter vanishes with the
step, and equal-order elements, which do not satisfy the inf-sup condition,
then lose their pressure stability [BochevGunzburgerLehoucq2007]_.  The code
therefore caps the weight at :math:`\rho \chi`, with

.. math::

   \chi = \min \left( \frac{b_0}{\Delta t},\ \frac{1}{\tau_P} \right) ,
   \qquad
   \tau_P = \left[ \left( \frac{2 |\tilde{\mathbf{v}}|}{h} \right)^2
   + 9 \left( \frac{4 \mu / \rho}{h^2} \right)^2 \right]^{-1/2} ,

with :math:`\tilde{\mathbf{v}}` the extrapolated velocity and :math:`\tau_P`,
in :math:`\mathrm{s}`, the product :math:`\rho \tau` of :eq:`flow-tau` without
its unsteady term.  It adds
:math:`\rho (b_0 / \Delta t - \chi)(\psi, \delta)` to the right side of
:eq:`flow-projection-pressure`, with
:math:`\delta = -\sum_j (b_j / b_0) \nabla \cdot \mathbf{v}^{n-j}`.  Where
:math:`\Delta t / b_0` exceeds :math:`\tau_P` the scheme is that of
[Fischer2021]_, and as :math:`\Delta t \to 0` it tends to the stabilized
discretization of the monolithic solver.  The added term multiplies a
divergence, so it leaves the exact solution unchanged.

**The solvers.**  The mass, stiffness and gradient matrices and the geometry of
every integration point are computed once.  The right-hand sides are assembled
at every step on several threads.  The pressure equation is solved with
algebraic multigrid (hypre BoomerAMG [FalgoutYang2002]_) and each Helmholtz
equation with a Jacobi preconditioner, inside the conjugate gradient method for
FEM, whose matrices are symmetric, and inside the generalized minimal residual
method (GMRES) for DMCDM, whose matrices are not symmetric in general.  Each
pressure solve starts from the combination of up to eight earlier pressures
that is closest to the solution [Fischer1998]_: in the energy norm for FEM, and
with the smallest residual for DMCDM.  The matrices and their preconditioners
are kept as long as :math:`b_0 / \Delta t` is unchanged.

**The time step.**  The ``"cfl"`` time stepper of ``solve_transient`` chooses
each step to keep the Courant number :math:`C = \Delta t \max_e
|\mathbf{v}|_e / h_e` at ``courant_number`` (default 0.5), with
:math:`|\mathbf{v}|_e` the largest speed at the nodes of element :math:`e` and
:math:`h_e` its size.

**Limits.**  The projection time integration needs the ``equal_order``
formulation, a positive density, Cartesian coordinates and a mesh without
hanging nodes.  The problem may hold the velocity and the pressure of the flow
alone, and the projection refuses any kernel or integrated boundary condition
on them that the flow did not add.  A body force is the ``body_force`` of
``fluid_dynamics``.

Buoyancy and natural convection
-------------------------------

When the density of a fluid depends on its temperature :math:`T`, the
temperature drives the flow and the flow carries the heat.  The Boussinesq
approximation keeps the density constant at :math:`\rho_0` except in the
gravity term, where :math:`\rho = \rho_0 [1 - \beta (T - T_0)]`.  A hydrostatic
pressure balances the constant part :math:`\rho_0 \mathbf{g}`, which leaves the
body force

.. math::
   :label: flow-boussinesq

   \mathbf{f} = -\rho_0\, \beta\, (T - T_0)\, \mathbf{g} ,

so fluid warmer than :math:`T_0` rises.  The kernel ``Boussinesq_buoyancy``
adds it to the equation of one velocity component, with the parameters
``temperature``, ``velocities``, ``gravity``, ``density`` (default 1),
``volumetric_expansion_coefficient`` and ``reference_temperature`` (default 0).
The ``nonisothermal_flow`` coupling adds it for every component with a non-zero
gravity when you give both ``gravity`` and ``volumetric_expansion_coefficient``,
with the density of the flow, and ``heat_convection`` in the energy equation
(:doc:`heat_transfer`).  The pressure of this coupling excludes the
hydrostatic part :math:`\rho_0 \mathbf{g} \cdot \mathbf{x}`.  In the input, a
problem with ``gravity``, ``heat_conduction`` and ``fluid_dynamics`` takes the
weight :math:`\rho \mathbf{g}` as a body force and the Boussinesq force on top
of it, with the properties ``volumetric_expansion_coefficient`` and
``reference_temperature``, so its pressure is the full pressure.

The buoyancy enters :math:`\mathbf{r}_m` of an equal-order flow with the same
load scaling as the force itself.  The coupling runs in both directions, and
the Jacobian holds the blocks :math:`\partial \mathbf{R}_{\mathbf{v}} / \partial
T` of the buoyancy and :math:`\partial R_T / \partial \mathbf{v}` of the
advected heat, with :math:`\mathbf{R}_{\mathbf{v}}` and :math:`R_T` the
residuals of the momentum and energy equations.  Automatic differentiation
computes both, so Newton's method converges quadratically on the coupled
problem.  The projection time integration does not take a temperature, so a
transient natural convection runs with the monolithic time integration.

With :math:`\kappa = k / (\rho_0 c_p)` the thermal diffusivity, :math:`k` the
thermal conductivity and :math:`c_p` the specific heat capacity, a cavity of
size :math:`L` with walls :math:`\Delta T` apart has the Rayleigh number
:math:`Ra = g \beta \Delta T L^3 / ((\mu / \rho_0) \kappa)` and the Prandtl
number :math:`Pr = (\mu / \rho_0) / \kappa`, with :math:`g = |\mathbf{g}|`.
Scaling the lengths by :math:`L`, the velocities by :math:`\kappa / L`, the
temperature differences by :math:`\Delta T` and the pressure by
:math:`\rho_0 \kappa^2 / L^2` gives the dimensionless equations with the
density, the conductivity and the heat capacity 1, the viscosity :math:`Pr`
and :math:`\rho_0 \beta g = Ra\, Pr`.  The average Nusselt number :math:`Nu`,
the heat flow through a heated wall divided by that of conduction alone, is
then the heat flow through the wall, the sum of the reactions of the
temperature equation there.  ``examples/natural_convection.py`` solves the
square cavity of [DeVahlDavis1983]_ in this form.

The verification cases of this page, with their sources, computed values and
errors, are on the :doc:`verification page <../verification>`.
