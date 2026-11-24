Heat transfer
=============

This page states the equations that the ``heat_transfer`` physics solves and
how the code discretizes them: conduction with a conductivity that can depend
on the temperature, volumetric heat sources, heat storage, the conditions on
the boundary and across a gap, and the heat that a flow carries.  Both methods
of the code assemble these equations: the finite element method (FEM) and the
dual mesh control domain method (DMCDM) [Reddy2024]_.  The input calls this
physics ``heat_conduction``, and the Python API calls it ``heat_transfer``.

The energy balance
------------------

Take a fixed region :math:`\omega` of a body, with boundary
:math:`\partial\omega` and outward unit normal :math:`\mathbf{n}`.  The thermal
energy stored in :math:`\omega` changes at the rate at which heat crosses its
boundary inwards plus the rate at which heat is generated inside it:

.. math::

   \frac{\mathrm{d}}{\mathrm{d}t} \int_\omega \rho\, c_p\, T \,\mathrm{d}V
   + \oint_{\partial\omega} \mathbf{q} \cdot \mathbf{n} \,\mathrm{d}S
   = \int_\omega s \,\mathrm{d}V .

Here :math:`T` is the temperature in :math:`\mathrm{K}`, :math:`\rho` the
density in :math:`\mathrm{kg/m^3}`, :math:`c_p` the specific heat capacity in
:math:`\mathrm{J/(kg\,K)}`, :math:`\mathbf{q}` the heat flux in
:math:`\mathrm{W/m^2}` and :math:`s` the volumetric heat source in
:math:`\mathrm{W/m^3}`.  Fourier's law states that heat flows down the
temperature gradient, :math:`\mathbf{q} = -k \nabla T`, with :math:`k > 0` the
thermal conductivity in :math:`\mathrm{W/(m\,K)}`.  The divergence theorem and
the arbitrary choice of :math:`\omega` give the energy equation

.. math::
   :label: heat-energy

   \rho\, c_p \frac{\partial T}{\partial t} + \nabla \cdot \mathbf{q} = s ,
   \qquad \mathbf{q} = -k \nabla T .

For a constant :math:`\rho c_p` it is the balance law
:math:`\partial m / \partial t + \nabla \cdot \mathbf{F} = s` with the storage
:math:`m = \rho c_p T` and the flux :math:`\mathbf{F} = \mathbf{q}`.

Three objects assemble the three terms of :eq:`heat-energy`, each with its own
coefficients:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Term
     - Object
     - Parameter of ``heat_transfer``
   * - flux :math:`\mathbf{q} = -k \nabla T`
     - ``heat_conduction``
     - ``thermal_conductivity``
   * - source :math:`s`
     - ``body_force``
     - ``heat_source``
   * - storage :math:`\rho c_p\, \partial T / \partial t`
     - ``heat_conduction_time_derivative``
     - ``density`` and ``specific_heat``

The storage term is discretized as
:math:`\rho c_p (T^{n+1} - T^n) / \Delta t`, with :math:`T^n` and
:math:`T^{n+1}` the temperatures at the start and the end of a step, and with
:math:`\rho c_p` evaluated at :math:`T^{n+1}`.  It is a time term: a steady
solve leaves it out, and the time-integration weight :math:`\theta` of
:doc:`nonlinear_and_time` multiplies the other terms only.  Only the product
:math:`\rho c_p` enters, so you can give a known volumetric heat capacity as
one of the two factors and set the other to 1.

The ``heat_transfer`` physics adds the storage term when you give both
``density`` and ``specific_heat``, and it refuses one without the other.  A
transient solve needs them.  When you leave ``thermal_conductivity`` out, the
physics reads the property ``thermal_conductivity`` of the property objects of
the problem.  The initial temperature, also the start of Newton's method in a
steady solve, defaults to :math:`293.15\ \mathrm{K}`.  In the input, the
``heat_conduction`` physics reads the properties ``thermal_conductivity`` and
``volumetric_heat_source``, and also ``density`` and ``specific_heat_capacity``
when the problem is transient or has a flow.

Temperature-dependent conductivity
----------------------------------

``heat_conduction`` multiplies a base conductivity :math:`k_0` by a
polynomial in the temperature:

.. math::
   :label: heat-kpoly

   k(\mathbf{x}, t, T) = k_0(\mathbf{x}, t)
   \left( c_0 + c_1 T + c_2 T^2 + \cdots \right) .

The base value :math:`k_0` is the parameter ``thermal_conductivity``: a
number, a function of :math:`(x, y, z, t)`, the name of a property, or an
expression of the fields.  The dimensionless coefficients
:math:`c_0, c_1, \dots` are the list ``temperature_polynomial``, with the
default :math:`\{1\}`, which leaves :math:`k_0` unchanged.  The list must
therefore carry its own constant term: a conductivity
:math:`k = 20\,(1 + 0.01\,T)\ \mathrm{W/(m\,K)}` is the base value 20 with
the list :math:`\{1, 0.01\}`.  The polynomial takes the temperature in the
units of the variable, with no reference temperature subtracted.

A conductivity that depends on :math:`T` makes :eq:`heat-energy` nonlinear.
Under Newton's method the polynomial takes the current iterate, and the
Jacobian holds the term :math:`(\partial k / \partial T) \nabla T`, so the
iteration converges quadratically.  Under Picard (direct) iteration the
polynomial takes the previous iterate as a constant, and the iteration
converges linearly.  :doc:`nonlinear_and_time` describes both iterations.

Boundary conditions
-------------------

Every point of the boundary carries one condition on the temperature or on the
normal heat flux :math:`q_n = \mathbf{q} \cdot \mathbf{n}`, the heat leaving
the body per unit area, in :math:`\mathrm{W/m^2}`.  A boundary without a
condition has :math:`q_n = 0`, so it is insulated.  The method
``add_boundary_condition`` of the physics takes a condition as text, with the
temperature or ``heat_flux`` (the heat leaving, :math:`q_n`) on the left, for
example ``"temperature = 40"`` or ``"heat_flux = 75*(temperature - 0)"``
(``tests/python/test_physics.py``).  The objects of the object level take the
opposite sign: ``Neumann`` and ``Robin`` prescribe :math:`-q_n`, the heat
entering the body.

Prescribed temperature
~~~~~~~~~~~~~~~~~~~~~~

The condition :math:`T = T_b(\mathbf{x}, t)`, with :math:`T_b` the prescribed
temperature in :math:`\mathrm{K}`, replaces the discrete equation of each node
on the boundary.  The residual of the replaced equation is the reaction there:
:meth:`dualmesh.physics.HeatTransfer.heat_flow` returns it as the heat leaving
through the boundary, in :math:`\mathrm{W}`.  The object is ``Dirichlet``.

Prescribed heat flux
~~~~~~~~~~~~~~~~~~~~

The condition prescribes :math:`q_n`, for example the heat of an electric
heater on the surface, with a negative value for heat entering.  A zero value
models an insulated surface or a plane of symmetry.  The input key is
``heat_flux`` and the object is ``Neumann``, whose parameter ``flux`` is
:math:`-q_n`.

Convection
~~~~~~~~~~

Newton's law of cooling lumps the thermal resistance of the fluid layer next
to the surface into one coefficient:

.. math::
   :label: heat-convection-bc

   q_n = h_c \left( T - T_\infty \right) ,

with :math:`h_c` the heat transfer coefficient in :math:`\mathrm{W/(m^2\,K)}`
and :math:`T_\infty` the temperature of the fluid far from the surface, in
:math:`\mathrm{K}`.  Heat leaves where the surface is hotter than the fluid.
Only the difference of temperatures enters, so any temperature scale works.
The condition is linear in :math:`T`, so a linear conduction problem stays
linear with it.  The input key is ``convection``, with the constants
``heat_transfer_coefficient`` and ``ambient_temperature``, as in
``examples/heated_plate.yaml``.  The object ``Robin`` prescribes
:math:`-q_n = q_0 - h_c (T - T_\infty)`, with ``transfer_coefficient``
:math:`h_c`, ``ambient_value`` :math:`T_\infty` and ``flux`` :math:`q_0`.  The
default ``ambient_value`` is 0, which cools the surface towards zero.

Radiation
~~~~~~~~~

The Stefan-Boltzmann law gives the heat that a grey surface radiates to large
surroundings:

.. math::
   :label: heat-radiation-bc

   q_n = \epsilon\, \sigma_{\mathrm{SB}} \left( T^4 - T_\infty^4 \right) ,

with :math:`\epsilon` the total hemispherical emissivity, dimensionless
between 0 and 1, :math:`\sigma_{\mathrm{SB}} = 5.670374419 \times 10^{-8}\
\mathrm{W/(m^2\,K^4)}` the Stefan-Boltzmann constant and :math:`T_\infty` the
temperature of the surroundings.  The fourth powers need an absolute
temperature scale: a problem in degrees Celsius gives a wrong result, and the
code cannot detect it.  The condition is strongly nonlinear.  Newton's method
differentiates the fourth power exactly.  Picard iteration factors the
difference as
:math:`T^4 - T_\infty^4 = (T^2 + T_\infty^2)(T + T_\infty)(T - T_\infty)` and
evaluates the first two factors at the previous iterate, which leaves a term
linear in :math:`T`.  The input key is ``radiation``, with the constants
``emissivity`` and ``ambient_temperature``.  The object is
``radiative_heat_flux``, with ``emissivity`` (default 1),
``stefan_boltzmann_constant`` (default :math:`5.670374419 \times 10^{-8}`) and
``ambient_temperature`` (default 0, radiation into deep space).

Heat transfer across a gap
~~~~~~~~~~~~~~~~~~~~~~~~~~

``gap_heat_transfer`` couples two bodies that face each other across a gap
without touching.  It pairs each integration point of the primary side set
``boundary`` with the closest point of ``secondary_boundary``, and it gives
the primary body the heat flux

.. math::
   :label: heat-gap

   q_g = h_g \left( T_s - T_p \right)
   + \sigma_{\mathrm{SB}}\, F_\epsilon \left( T_s^4 - T_p^4 \right) ,
   \qquad F_\epsilon = \left( \frac{1}{\epsilon_p} + \frac{1}{\epsilon_s} - 1
   \right)^{-1} ,

with :math:`q_g` the heat entering the primary body per unit primary area, in
:math:`\mathrm{W/m^2}`, :math:`T_p` and :math:`T_s` the temperatures of the
primary and the secondary surface, :math:`h_g` the gap conductance
(``gap_conductance``, default 0) in :math:`\mathrm{W/(m^2\,K)}`, and
:math:`F_\epsilon` the radiation exchange factor of two parallel grey surfaces
of emissivities :math:`\epsilon_p` and :math:`\epsilon_s`
(``primary_emissivity`` and ``secondary_emissivity``, default 0).  The
radiation term enters when both emissivities are positive, and the
temperatures must then be absolute.  The secondary body receives the same heat,
so the gap conserves energy.

A deforming body
~~~~~~~~~~~~~~~~

When ``heat_conduction`` takes the parameter
``deformation_gradient_property``, the heat conducts through the deformed body.
With :math:`\mathbf{F}_d` the deformation gradient, :math:`J = \det
\mathbf{F}_d` and :math:`\nabla_X` the gradient on the undeformed mesh, the
flux per unit undeformed area is
:math:`-J k\, \mathbf{F}_d^{-1} \mathbf{F}_d^{-\mathsf{T}} \nabla_X T`.  The
boundary conditions with the same parameter multiply their flux, given per
unit deformed area, by the area ratio
:math:`J |\mathbf{F}_d^{-\mathsf{T}} \mathbf{N}|`, with :math:`\mathbf{N}` the
unit normal of the undeformed surface.

The heat carried by a flow
--------------------------

In a fluid that moves with the velocity :math:`\mathbf{v}`, the heat is carried
with it, and the energy equation gains an advective term:

.. math::
   :label: heat-advection

   \rho\, c_p \left( \frac{\partial T}{\partial t}
   + \mathbf{v} \cdot \nabla T \right) + \nabla \cdot \mathbf{q} = s .

For an incompressible flow, with :math:`\nabla \cdot \mathbf{v} = 0`, this is
the balance law with the storage :math:`m = \rho c_p T` and the flux
:math:`\mathbf{F} = \rho c_p T \mathbf{v} + \mathbf{q}`, because
:math:`\nabla \cdot (T \mathbf{v}) = \mathbf{v} \cdot \nabla T + T \nabla
\cdot \mathbf{v}`.  A discrete velocity is not exactly divergence free, so the
two forms differ by :math:`\rho c_p T \nabla \cdot \mathbf{v}`, a term that
depends on the temperature scale.

``heat_convection`` assembles the advective form.  In an equal-order flow it
carries the velocity :math:`\mathbf{w}` whose flux the stabilized mass equation
conserves (:ref:`the conserved velocity <conserved-advection>`), and in the
penalty and Taylor-Hood formulations it carries :math:`\mathbf{v}`.  FEM
integrates :math:`\rho c_p N_i\, \mathbf{w} \cdot \nabla T` over the domain for
the equation of node :math:`i`.  DMCDM sums
:math:`\rho c_p (\mathbf{w} \cdot \mathbf{A}_f)(T_f - T_i)` over the faces
:math:`f` of the control domain :math:`V_i`, with :math:`\mathbf{A}_f` the
outward area vector of the face, :math:`T_f` the face value of the advection
scheme and :math:`\mathbf{v}` on the boundary of the domain
(:doc:`finite_volume`).  Two properties follow from the mass equation of an
equal-order flow, which holds for every test function of the temperature
because the pressure is interpolated like it:

* The heat is conserved exactly: summed over all nodes, the advective term is
  the heat that the discrete mass flux carries out of the domain.
* The temperature does not depend on its scale: the term has no part
  :math:`T \nabla \cdot \mathbf{w}`, so a constant added to every temperature
  changes no equation.

The ``nonisothermal_flow`` coupling of a ``fluid_dynamics`` and a
``heat_transfer`` physics adds ``heat_convection``, with the density of the
flow (1 for a Stokes flow, which has no density) and the specific heat of the
heat transfer (1 when it is not given).  In the input, a problem with
``heat_conduction`` and ``fluid_dynamics`` takes this coupling.  The Jacobian
of the coupled problem holds the derivatives of the advective term with respect
to the velocity, so Newton's method converges quadratically on it.

Advection schemes
~~~~~~~~~~~~~~~~~

The parameter ``advection_scheme`` of ``heat_transfer`` chooses how the heat
that a flow carries is discretized:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Value
     - Methods
     - Discretization
   * - ``"SUPG"`` (default)
     - FEM, DMCDM
     - central term plus the streamline upwind Petrov-Galerkin (SUPG)
       stabilization below
   * - ``"central"``
     - FEM, DMCDM
     - central value of the temperature, with no stabilization
   * - ``"upwind"``
     - DMCDM
     - upstream face value, first order and bounded
   * - ``"linear_upwind"``
     - DMCDM
     - upstream face value corrected by the upstream gradient, second order
   * - ``"limited"``
     - DMCDM
     - upstream face value blended towards the central one by the van Albada
       limiter, second order and bounded

:doc:`finite_volume` defines the face values.  The best scheme depends on the
case: where conduction matters as much as convection, as in resolved natural
convection, ``"central"`` can give the smaller error.

The streamline upwind Petrov-Galerkin stabilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cell Péclet number :math:`Pe_h = |\mathbf{v}|\, h / (2 \kappa)`, with
:math:`h` the element size and :math:`\kappa = k / (\rho c_p)` the thermal
diffusivity, compares convection with conduction across one element.  At a
large :math:`Pe_h` the central discretization oscillates.  The SUPG method of
[BrooksHughes1982]_ adds a term that acts along the streamlines and vanishes
for the exact solution.  ``heat_streamline_stabilization`` assembles it as the
flux :math:`-\tau\, r\, \mathbf{v}`, which adds

.. math::

   \int_\Omega \tau\, (\mathbf{v} \cdot \nabla \psi)\, r \,\mathrm{d}V

to the equation of the test function :math:`\psi`.  FEM integrates it over the
elements, and DMCDM integrates the flux over the faces of the control domains.
The residual :math:`r` of the energy equation, in :math:`\mathrm{W/m^3}`, is

.. math::
   :label: heat-supg-residual

   r = \rho\, c_p \left( \frac{\partial T}{\partial t}
   + \mathbf{w} \cdot \nabla T \right) - k\, \nabla^2 T - s .

The kernel builds :math:`r` as the sum of the sources of every other kernel of
the temperature, so a heat source added with ``add_kernel`` enters it too.  It
refuses an ``advection`` kernel of the temperature written as a flux, whose
term a sum of sources cannot hold.  Quadratic elements give the conduction term
:math:`k \nabla^2 T` from the second derivatives of their shape functions, with
the gradient of :math:`k` left out.  Linear elements leave the term out,
because it vanishes inside them.

The parameter :math:`\tau`, in :math:`\mathrm{s}`, is that of [Tezduyar1991]_:

.. math::
   :label: heat-tau

   \tau = \left[ \left( \frac{2 |\mathbf{v}|}{h} \right)^2
   + 9 \left( \frac{4 \kappa}{h^2} \right)^2
   + \left( \frac{2}{\Delta t} \right)^2 \right]^{-1/2} ,

with the last term in a transient solve only.  Its limits are
:math:`h / (2 |\mathbf{v}|)` where convection dominates and
:math:`h^2 / (12 \kappa)` where conduction dominates.  The second is the limit
of the optimal parameter
:math:`(h / 2|\mathbf{v}|)(\coth Pe_h - 1/Pe_h)` of linear elements in one
dimension [BrooksHughes1982]_, which the factor 9 gives.  The element size is
:math:`h = (c V_e)^{1/d}`, with :math:`V_e` the volume of the element,
:math:`d` the dimension of the mesh and :math:`c = 1` for quadrilaterals and
hexahedra, 2 for triangles and prisms, and 6 for tetrahedra and pyramids, so
that each element of a structured grid of spacing :math:`h` has the size
:math:`h`.  A quadratic element has half that size.

On quadratic elements, and for a conductivity uniform in space, the residual of
the exact solution is zero, and the stabilization is consistent.  On linear
elements the residual of the exact solution is :math:`k \nabla^2 T`, so the
stabilization adds an error of the order of :math:`\tau k |\nabla^2 T|`.  This
error is small where convection dominates, and it is zero for a harmonic
temperature, with :math:`\nabla^2 T = 0`.
SUPG does not bound the temperature, so it can overshoot at a sharp layer.  The
bounded schemes are ``"upwind"`` and ``"limited"``.

In a step of the theta method of :doc:`nonlinear_and_time`, the time
derivative inside :math:`r` enters with the weight 1, whatever the weight
:math:`\theta` of the steady terms.  The explicit step (``forward_Euler``,
:math:`\theta = 0`) cannot weight it, so it refuses this kernel.

The verification cases of this page, with their sources, computed values and
errors, are on the :doc:`verification page <../verification>`.
