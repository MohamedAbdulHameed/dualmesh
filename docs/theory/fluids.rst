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

* ``"stress"`` (the default, except with the projection time integration)
  takes the stress above.  Its natural boundary
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
positive density, ``advection`` (the inertia :math:`\rho (\mathbf{v} \cdot
\nabla) v_i`, the one transport kernel of the code with the coefficient
:math:`\rho`), ``momentum_stabilization`` and ``time_derivative`` (the term
:math:`\rho\, \partial v_i / \partial t` of a transient solve, the one storage
kernel with the coefficient :math:`\rho`).  A ``body_force`` adds :math:`f_i`.  The mass equation is the
kernel ``mass_conservation`` on the pressure variable, and the boundary
condition ``mass_flux`` on every side set closes it.  ``mass_conservation``
holds the pressure stabilization below unless the pressure and the velocity form
the Taylor-Hood element, which needs none.

The stabilized mass flux
~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\mathbf{r}_m`, in :math:`\mathrm{N/m^3}`, be the residual of the
momentum equation in its strong form,

.. math::
   :label: flow-residual

   \mathbf{r}_m = \rho_t \frac{\partial \mathbf{v}}{\partial t}
   + \rho (\mathbf{v} \cdot \nabla) \mathbf{v} + \nabla p
   - \mu \nabla^2 \mathbf{v} - \mathbf{f} ,

with :math:`\rho_t` the coefficient of the ``time_derivative`` and :math:`\rho`
that of the ``advection`` of the velocity, both the density of the flow.

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

   \tau = \left[ \left( \frac{2 \rho_t}{\Delta t} \right)^2
   + \left( \frac{2 \rho |\mathbf{v}|}{h} \right)^2
   + 9 \left( \frac{4 \mu}{h^2} \right)^2 \right]^{-1/2} .

The three terms are the reciprocal time scales of the unsteadiness, of the
convection across an element and of the diffusion across it, each times
:math:`\rho`.  The first term enters a transient solve only.  The code computes
:math:`\tau` with the one function that also gives the parameter of the
streamline stabilization of the heat that a flow carries
(:doc:`heat_transfer`), and the cap of the pressure stabilization of the
projection time integration below.  In the steady
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
``mass_conservation`` and ``momentum_stabilization`` build :math:`\mathbf{r}_m`
from the kernels of the velocities, each of which gives its term of the strong
residual: the storage of ``time_derivative``, the inertia of ``advection`` with
the velocity :math:`\mathbf{v}`, the pressure gradient of
``pressure_gradient``, the viscous term of ``viscous_stress`` and the sources
of ``body_force``, ``Boussinesq_buoyancy`` and the other kernels.  The heat
stabilization builds its residual in the same way.  They refuse a kernel on a
velocity with a flux whose divergence :math:`\mathbf{r}_m` does not hold.  In
a step of the theta method with :math:`\theta < 1`, the steady terms enter
:math:`\mathbf{r}_m` at the old state too, which the fields, the position and
the time give.  A term that reads the properties of the material at the old
state, which the step does not keep, makes the stabilized flow take the
``backward_Euler`` scheme.

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
:math:`\tau \mathbf{r}_m`.  ``advection`` of the momentum and of the heat
therefore carries :math:`\mathbf{w}` in an equal-order flow, in the advective
form.  FEM integrates
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
object ``advection`` takes the parameter ``advection_scheme`` for a flow
assembled object by object:

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
unknown nearest the centre of all free pressure unknowns at its value, with a
tie to the node of the smaller number, so that a distributed run holds the node
of a serial one.  After each converged solve, it shifts the pressure to a mean
of zero over the fluid and takes the reactions at the shifted state.  The mean
weighs each node as the lumped rule does: by the measure of its control domain
under DMCDM and by the integral of its shape function under FEM, times the
coordinate factor at the node.  When a viscosity, a density or a source depends on the
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

The start of a transient flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial velocity of an incompressible flow must satisfy the discrete mass
balance, with its normal component equal to the prescribed velocity on the
boundary.  A fluid at rest behind a prescribed inflow does not.  Its first step
then has to remove a divergence that does not shrink with the step size, and
the error control of the time step refuses every step.  A transient solve
therefore first projects the initial velocity onto a field without divergence.

Let :math:`\mathbf{u}_0` be the vector of the initial nodal velocities, which
holds the values prescribed at the start time, and :math:`\mathbf{M}` the mass
matrix of the time derivative of the velocity, divided by the density.  The
discrete divergence :math:`\mathbf{D}` is that of the mass balance without its
stabilization, assembled by the problem from ``mass_conservation`` and
``mass_flux``.  Its row of node :math:`i` is
:math:`(\psi_i, \nabla \cdot \mathbf{v})` for FEM, with the shape function
:math:`\psi_i` of the pressure and the coordinate factor, or the flux
:math:`\oint_{\partial V_i} \mathbf{v} \cdot \mathbf{n} \,\mathrm{d}S` out of
the control domain :math:`V_i` for DMCDM.  The penalty formulation, which has
no pressure, tests the balance with the functions of the first velocity
component.  A flow without a time derivative of its velocity, a Stokes flow,
has no initial velocity of its own, and the start leaves it as it is.
The projected velocity :math:`\mathbf{u} = \mathbf{u}_0 + \delta\mathbf{u}`
satisfies :math:`\mathbf{D} \mathbf{u} = 0`, with :math:`\delta\mathbf{u}`
zero at every prescribed value.  For FEM it is the field nearest to
:math:`\mathbf{u}_0` in the norm of :math:`\mathbf{M}`.  With a multiplier
:math:`\boldsymbol{\lambda}` of one value per node, the correction is
:math:`\delta\mathbf{u} = -\mathbf{M}^{-1} \mathbf{D}^{\mathsf{T}}
\boldsymbol{\lambda}`, and :math:`\boldsymbol{\lambda}` solves the Poisson
problem :math:`\mathbf{D} \mathbf{M}^{-1} \mathbf{D}^{\mathsf{T}}
\boldsymbol{\lambda} = \mathbf{D} \mathbf{u}_0`.  The code solves it in the
saddle point form

.. math::
   :label: flow-start-projection

   \begin{bmatrix} \mathbf{M} & \mathbf{D}^{\mathsf{T}} \\
   \mathbf{D} & -\epsilon \mathbf{K}_\lambda \end{bmatrix}
   \begin{bmatrix} \delta\mathbf{u} \\ \boldsymbol{\lambda} \end{bmatrix}
   = \begin{bmatrix} \mathbf{0} \\ -\mathbf{D} \mathbf{u}_0 \end{bmatrix} ,

with a direct factorization.  :math:`\mathbf{K}_\lambda` is the Laplacian
:math:`(\nabla \psi_i, \nabla \psi_j)` of the multiplier, that of a
``diffusion`` kernel, whose spectrum is
that of :math:`\mathbf{D} \mathbf{M}^{-1} \mathbf{D}^{\mathsf{T}}` on smooth
fields, and :math:`\epsilon = 10^{-4}`.  The spurious pressure modes of
equal-order elements are multipliers that :math:`\mathbf{D}^{\mathsf{T}}` maps
to zero or nearly to zero, so no velocity of bounded size changes their part of
the divergence.  The regularization :math:`\epsilon \mathbf{K}_\lambda` fixes
the multiplier on these modes.  Corrections with the residual of the system
without it then remove its effect from the other modes, until a correction
changes the velocity by less than :math:`10^{-13}` of the speed.  The
multiplier is zero where the pressure is prescribed, since the boundary
condition replaces the mass balance there.  A flow whose normal velocity is
prescribed on the whole boundary has its multiplier fixed at one node, and a
net flux of the prescribed velocities spreads over its volume as a uniform
divergence.

A field without divergence leaves the projection unchanged.  For a field with
:math:`\mathbf{D} \mathbf{u} = 0`, a small step changes the velocity in
proportion to its size, so the error control starts with the steps that the
flow itself needs.  A run that restarts from a checkpoint, or that starts from
the state that the last transient run of the same problem reached, keeps that
state as it is.  The report of the solve gives the size of the correction: the
largest change of the velocity at a node over the largest speed after the
projection.

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
point system by a pressure Poisson problem and the problems of the velocity,
each with its own fast solver.  Its operators come from the problem's own
assembly of the kernels of the flow, so a step has the coordinates, the
formulation (equal-order or Taylor-Hood), the viscous form, the hanging nodes,
the periodic boundaries and the numbering of the monolithic solve.  Let
:math:`\ell` be the ``time_order`` (1, 2 or 3, default 2),
:math:`b_0, \dots, b_\ell` the dimensionless coefficients of the backward
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
   = \frac{\rho}{\Delta t} \mathbf{v}^* - \nabla p^n
   + \nabla \cdot \boldsymbol{\tau}(\mathbf{v}^n) ,

with :math:`\boldsymbol{\tau}` the viscous stress of ``viscous_form``:
:math:`\mu (\nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}})` or
:math:`\mu \nabla \mathbf{v}`.  The coefficients come from the time levels
themselves: :math:`b_j / \Delta t` is the derivative at :math:`t^n` of the
Lagrange polynomial of :math:`t^{n-j}` on the levels
:math:`t^n, \dots, t^{n-\ell}`, and :math:`a_j` the value at :math:`t^n` of
the Lagrange polynomial of :math:`t^{n-j}` on the levels
:math:`t^{n-1}, \dots, t^{n-\ell}`.  A step shortened to land on an output
time therefore keeps the order.  The first steps take the orders below
:math:`\ell`.

**The terms of the momentum equations.**  The step reads the role of every
kernel of the momentum equations.  The time derivative gives the mass matrix
:math:`\mathbf{M}` with the density.  The terms written as a flux, the viscous
stress and the pressure, are taken at :math:`t^n` through their Jacobian at
rest, which holds for terms linear in the flow: the viscous operator
:math:`\mathbf{A}` and the pressure gradient :math:`\mathbf{G}`.  The inertia,
the advection of the velocity, is extrapolated in the non-conservative form of
:eq:`flow-projection-star`.  Every other source, a body force, is taken at
:math:`t^n` with the extrapolated velocity, and the momentum stabilization of
the monolithic solve is left out, since the splitting stabilizes the pressure
itself.  The density :math:`\rho` and the viscosity :math:`\mu` are those of
the kernels.

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
over the boundary for a divergence-free :math:`\mathbf{v}^n`.  The second is the
viscous term in its rotational form :math:`-\mu \nabla \times \boldsymbol{\omega}`,
which differs from :math:`\mu \nabla^2 \mathbf{v}` by
:math:`\mu \nabla (\nabla \cdot \mathbf{v})`.  The pressure then has the
boundary condition of the momentum equation without the non-physical condition
:math:`\partial p / \partial n = 0` of the standard form and its numerical
boundary layer.  The scheme is the velocity-correction scheme in rotational form
of [GuermondShen2003]_, which they show to be the splitting of
[KarniadakisIsraeliOrszag1991]_.  It needs only first derivatives of the
velocity, so it applies to linear elements.  On the Taylor-Green vortex with
walls on 16 by 16 quadrilaterals, the difference of the solution at
:math:`t = 0.5` from that of the step :math:`\Delta t = 0.5/512` is 3 to 6
4 to 8 times smaller in the pressure, and 2.7 to 3.8 times smaller in the
velocity, than with the standard form, which leaves the viscous term out of the pressure equation
on linear elements (FEM and DMCDM, steps 0.1 to 0.0125).  The source
:math:`\mathbf{f}^n / \rho` enters :math:`\mathbf{v}^*` as its projection onto
the velocity field, the solution of :math:`\mathbf{M} \mathbf{s} = \mathbf{F}`
for the vector :math:`\mathbf{F}` of the source kernels.  Where a condition
prescribes the pressure, the equation takes that value.  An enclosed flow has
its pressure fixed at one node after the right-hand side is made to sum to
zero, and then shifted to a mean of zero.

**The velocity.**  The velocity then solves the momentum equations at
:math:`t^n`,

.. math::
   :label: flow-projection-helmholtz

   \frac{b_0}{\Delta t} \mathbf{M} \mathbf{v}^n + \mathbf{A} \mathbf{v}^n
   = -\frac{1}{\Delta t} \mathbf{M} \sum_{j=1}^{\ell} b_j \mathbf{v}^{n-j}
   - \rho \left( \psi, \sum_{j=1}^{\ell} a_j (\mathbf{v}^{n-j} \cdot \nabla)
   \mathbf{v}^{n-j} \right) + \mathbf{F}^n - \mathbf{G} p^n ,

with the prescribed velocity on its boundary.  :math:`\mathbf{G}` is the
pressure term in its flux form :math:`-(\nabla \cdot \psi, p)`, so a boundary
where the velocity is free has the natural condition of the viscous form with
the pressure there, as in the monolithic solve: the do-nothing outflow
:math:`\mu\, \partial \mathbf{v} / \partial n - p \mathbf{n} = \mathbf{0}`, or
zero traction.  The ``"laplacian"`` form leaves the components apart, and each
solves a Helmholtz equation of its own.  The ``"stress"`` form couples them in
one system.  The step keeps the residual of the momentum equations, so the
reactions and the force of the fluid on a wall are available as after a
monolithic step.

**The two methods.**  FEM takes the shape functions of the pressure as the test
functions :math:`\psi` of the pressure equation, the linear ones of the corners
for the Taylor-Hood element.  DMCDM takes the indicator of the control domain
:math:`V_i` of each node, so that :math:`(\psi, g)` becomes
:math:`\int_{V_i} g \,\mathrm{d}V` and :math:`(\nabla \psi, \mathbf{G})`
becomes :math:`-\oint_{\partial V_i} \mathbf{G} \cdot \mathbf{n}
\,\mathrm{d}S` over the interfaces between control domains, for a scalar
:math:`g` and a vector :math:`\mathbf{G}`.  The Laplacian of the pressure is
that of a ``diffusion`` kernel on the pressure: the flux of the gradient out of
the control domains for DMCDM.  The curl-curl term keeps the shape functions.
The operators take the rules of their kernels.  The right-hand sides take, for
FEM, a Gauss rule of one point more per direction than the order of the element,
and for DMCDM the midpoint rule of every part of a control domain and of every
interface on linear elements.  Every integral carries the coordinate factor,
:math:`2 \pi r` in axisymmetric coordinates, where the divergence holds
:math:`v_r / r`.

**Hanging nodes and periodic boundaries.**  The step solves in the unknowns of
the problem.  A dependent value, at a node hanging under FEM or at a periodic
copy, is the weighted sum of independent ones: every matrix :math:`\mathbf{A}`
becomes :math:`\mathbf{C}^{\mathsf{T}}\mathbf{A}\,\mathbf{C}` and every
right-hand side :math:`\mathbf{b}` becomes :math:`\mathbf{C}^{\mathsf{T}}
\mathbf{b}`, with :math:`\mathbf{C}` the matrix that maps the independent
values to all of them, and the dependent values follow each solve.  DMCDM keeps
a hanging node as an unknown with a control domain of its own: a large cell
with nodes hanging on its sides is split as a polygon, whose control domains
carry the integrals and whose interfaces carry the fluxes.

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
divergence, so it leaves the exact solution unchanged.  The Taylor-Hood element
is stable without it and keeps the weight :math:`\rho b_0 / \Delta t`.  For
DMCDM the part :math:`\rho (b_0 / \Delta t)(\psi, \delta)` is the flux of
:math:`-\sum_j (b_j / b_0) \mathbf{v}^{n-j}` out of the control domain, as
:math:`(\nabla \psi, \mathbf{v}^*)` is.  The terms in
:math:`\rho b_0 / \Delta t` then cancel inside the domain for every velocity
of the earlier steps, also where the velocity jumps across the side of a cell
with hanging nodes.

**The steady state.**  At a steady state the terms in :math:`b_0 / \Delta t`
of the two equations cancel, and the pressure equation states
:math:`\chi (\psi, \nabla \cdot \mathbf{v})` against the residual of the
momentum equations: the steady state is that of a pressure stabilization of
parameter :math:`1 / \chi`.  Below :math:`\Delta t = b_0 \tau_P` it is
:math:`\tau_P` whatever the step, and the steady state does not depend on the
step.  Above, the parameter is :math:`\Delta t / b_0`, and the steady state
depends on the step that reaches it.  The rotational form does not change
this, since it acts on the boundary condition of the pressure, while the
dependence comes from the weight of the divergence.  The error control of the
time step lets the step grow as the flow settles, so a flow marched to its
steady state ends with a step above the cap.  The discrete velocity then has a
divergence that grows with the step, and the advection, which carries this
velocity in the non-conservative form, no longer conserves the momentum
exactly.  A body force of :math:`3\ \mathrm{N/m^3}` that drives the flow of
unit density and viscosity through a channel of 8 by 6 triangles, periodic in
:math:`x`, puts :math:`6\ \mathrm{N/m}` on the walls at the steady state.
Marched to :math:`t = 4\ \mathrm{s}` by the projection, the force on the walls
has the relative error :math:`2.4 \times 10^{-7}` for every fixed step up to
:math:`4 \times 10^{-3}\ \mathrm{s}`, and :math:`4.6 \times 10^{-7}`,
:math:`9.7 \times 10^{-7}` and :math:`1.4 \times 10^{-6}` with the steps
0.0125, 0.05 and :math:`0.2\ \mathrm{s}` (FEM and DMCDM alike).  With the error
control, whose last step is :math:`0.58\ \mathrm{s}` for FEM and
:math:`1.2\ \mathrm{s}` for DMCDM, it is :math:`1.9 \times 10^{-6}` and
:math:`2.0 \times 10^{-6}`.  The monolithic solve, whose
advection carries the velocity that its mass equation conserves
[HughesWells2005]_, has no splitting, so it computes a steady state best.

**The solvers.**  The mass, viscous, gradient and pressure matrices and the
geometry of every integration point of the right-hand sides are computed once.
The right-hand sides are assembled at every step on several threads.  The
pressure equation is solved with algebraic multigrid (hypre BoomerAMG
[FalgoutYang2002]_) and the velocity equations with a Jacobi preconditioner,
inside the conjugate gradient method for FEM, whose matrices are symmetric, and
inside the generalized minimal residual method (GMRES) for DMCDM, whose
matrices are not symmetric in general.  Each pressure solve starts from the
combination of up to eight earlier pressures that is closest to the solution
[Fischer1998]_: in the energy norm for FEM, and with the smallest residual for
DMCDM.  The matrices and their preconditioners are kept as long as
:math:`b_0 / \Delta t` is unchanged.  A source that does not change is
projected once.

**The time step.**  The default ``"error"`` time stepper of
``solve_transient`` compares each step with the polynomial through the last
accepted states, extrapolated to the end of the step
(:doc:`nonlinear_and_time`).  The ``"cfl"`` time stepper chooses
each step to keep the Courant number :math:`C = \Delta t \max_e
|\mathbf{v}|_e / h_e` at ``courant_number`` (default 0.5), with
:math:`|\mathbf{v}|_e` the largest speed at the nodes of element :math:`e` and
:math:`h_e` its size.

**Limits.**  The projection time integration needs a pressure, so it refuses
the penalty formulation, and a fluid of one positive density.  The problem
holds the velocity and the pressure of the flow alone.  The viscous and
pressure operators are assembled at the start of the run, so a viscosity that
changes in time keeps its value at the start.  A boundary without a prescribed
velocity needs a prescribed pressure, which the pressure equation takes as its
condition there, and a traction on the boundary of the momentum equations is
refused.  The default viscous form is ``"laplacian"``, whose difference from
the stress form, :math:`\mu \nabla (\nabla \cdot \mathbf{v})`, the rotational
pressure leaves out.  With ``"stress"`` the step takes that term at the new
time, where the splitting leaves the velocity a divergence, which costs
pressure accuracy: on the Taylor-Green vortex with walls the pressure error
falls by 7.0 instead of 8.7 from 8 to 32 quadrilaterals with FEM, and by 5.5
instead of 7.4 with DMCDM.  The open channel of
``test_the_projection_has_the_outflow_of_its_viscous_form`` reaches, with
either form, the outflow of the steady solve of that form.

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
with the density of the flow, and ``advection`` of the heat in the energy equation
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
temperature equation there.  The tests in
``tests/python/test_multiphysics.py`` solve the square cavity of
[DeVahlDavis1983]_ in this form, and ``examples/natural_convection`` solves it
for air in SI units.

The verification cases of this page, with their sources, computed values and
errors, are on the :doc:`verification page <../verification>`.
