Heat transfer and incompressible flow
=====================================

This chapter sets out the two continuum models implemented by the
``heat_transfer`` and ``fluids`` modules: conduction of heat in a solid, with a
conductivity that may depend on the temperature, and the slow flow of a viscous
incompressible fluid.  Both are developed from the underlying balance law,
because the library never asks for a weak form or for element matrices.  It asks
for the governing equation written in the canonical conservation form

.. math::
   :label: canonical_hf

   \mathcal{R}(u) \equiv -\nabla \cdot \mathbf{F}(u, \nabla u, \mathbf{x}, t)
   + S(u, \nabla u, \mathbf{x}, t) = 0 ,

in which :math:`\mathbf{F}` is the flux of the conserved quantity and :math:`S`
collects everything that is not a divergence.  A *kernel* supplies
:math:`\mathbf{F}`, or :math:`S`, or both, and the discretizations consume that
one statement: the dual mesh control domain method integrates it over the
control domain of each node and turns the divergence into a contour integral,
while the Galerkin finite element method multiplies it by a test function and
integrates by parts.  Every equation below is therefore presented as a pair
:math:`(\mathbf{F}, S)`, and the pairs given here are the ones the code
assembles.

Heat conduction
---------------

The energy equation
^^^^^^^^^^^^^^^^^^^

Take any fixed region :math:`\omega` inside the body, with boundary
:math:`\partial\omega` and outward unit normal :math:`\mathbf{n}`.  Conservation
of energy states that the stored thermal energy of :math:`\omega` grows at the
rate at which heat crosses into it plus the rate at which heat is generated
inside it:

.. math::

   \frac{\mathrm{d}}{\mathrm{d}t} \int_\omega \rho\, c_p\, T \, \mathrm{d}V
   = - \oint_{\partial\omega} \mathbf{q} \cdot \mathbf{n} \, \mathrm{d}S
   + \int_\omega q''' \, \mathrm{d}V .

Here :math:`T` is the temperature, :math:`\rho` the mass density, :math:`c_p`
the specific heat capacity (so :math:`\rho c_p` is the energy stored per unit
volume per degree), :math:`\mathbf{q}` the heat flux vector and :math:`q'''` the
volumetric generation rate.  The minus sign on the surface integral is there
because :math:`\mathbf{q} \cdot \mathbf{n}` measures heat *leaving*, whereas the
balance needs heat entering.  Fourier's law adds the constitutive statement that
heat flows down the temperature gradient,
:math:`\mathbf{q} = -k \, \nabla T`, with :math:`k > 0` the thermal
conductivity.  Substituting it, converting the surface integral by the
divergence theorem and using the fact that :math:`\omega` was arbitrary gives
the pointwise energy equation

.. math::
   :label: energy

   \rho \, c_p \, \frac{\partial T}{\partial t}
   - \nabla \cdot \left( k \, \nabla T \right) = q''' .

Comparison with :eq:`canonical_hf` identifies the flux
:math:`\mathbf{F} = k \nabla T`, which is the negative of the physical heat flux
vector, and the source
:math:`S = \rho c_p \, \partial T / \partial t - q'''`.  The code splits this
across three kernels so that each piece carries its own coefficients and its own
quadrature rule.  ``heat_conduction`` supplies the flux :math:`k \nabla T` and no
source.  ``heat_source`` supplies no flux and the source :math:`S = -q'''`, the
sign being what makes a positive generation rate add heat to the body.
``heat_conduction_time_derivative`` supplies no flux and the discrete capacity term
:math:`S = \rho c_p (T - T_{\text{old}}) / \Delta t`, with
:math:`T_{\text{old}}` the temperature at the start of the step.  That kernel is
marked as a *time kernel*, with two consequences worth knowing: it is omitted
entirely from a steady solve, and it is not multiplied by the time-integration
weight :math:`\theta`, which applies to the steady terms alone.  Only the
product :math:`\rho c_p` enters, so a volumetric heat capacity known as one
number is supplied by leaving the other factor at unity.

The sign of the natural boundary quantity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :eq:`canonical_hf` is integrated over a control domain touching the
boundary, the divergence theorem leaves behind the normal component of the flux,

.. math::

   q_n \equiv \mathbf{n} \cdot \mathbf{F} = \mathbf{n} \cdot (k \nabla T)
           = - \mathbf{n} \cdot \mathbf{q} ,

on the part of the boundary belonging to that control domain.  This is the
*secondary variable* of the energy equation and the quantity an integrated
boundary condition prescribes.  Read the equality from right to left: because
:math:`\mathbf{n}` points outwards, :math:`\mathbf{n} \cdot \mathbf{q}` is heat
leaving, so :math:`q_n` is **the heat flux entering the body**, in watts per
square metre, counted positive inwards.  The convention holds without exception
across the library, in this module and in the reactions recovered at constrained
nodes, and it is worth stating plainly because the opposite convention is at
least as common elsewhere.  A positive prescribed flux heats the body, and a
reported reaction of :math:`+4817\ \mathrm{W}` at a fixed boundary means that
much power flows inwards through it.

The convention also explains behaviour that surprises some users: a boundary
carrying no condition at all is insulated.  Nothing special is done to arrange
this.  The term :math:`q_n` appears in the discrete equation of every boundary
node, and if no object writes a value into it, it contributes nothing, which is
the statement :math:`q_n = 0`.  Zero heat flux is an adiabatic surface, so a
boundary left out of the input is not an error and not an open boundary but a
perfectly insulated one.

Boundary conditions
^^^^^^^^^^^^^^^^^^^

Exactly one member of the pair :math:`(T, q_n)` is specified at every boundary
point, and the module offers four ways to do it.

The **prescribed temperature**, or essential, condition sets
:math:`T = g(\mathbf{x}, t)` and is imposed by replacing the discrete equation
of each affected node with that statement, the displaced equation being retained
so that the secondary variable can be recovered afterwards as a reaction.  Use
it wherever the surface temperature is controlled or known.  The object is
``Dirichlet_boundary_condition``.

The **prescribed flux**, or natural, condition sets
:math:`\mathbf{n} \cdot (k \nabla T) = q(\mathbf{x}, t)`, with :math:`q` the
heat entering per unit area.  Use it for a surface heated by a known source such
as an electrical element, and use its default value of zero, or no condition at
all, for an insulated or symmetry surface.  The object is ``heat_flux_boundary_condition``, the
heat-transfer spelling of the generic ``Neumann_boundary_condition``, carrying the same sign
convention.

The **convection** condition applies Newton's law of cooling, which models the
thermal resistance of the fluid boundary layer washing over the surface by one
lumped coefficient:

.. math::
   :label: convection

   \mathbf{n} \cdot (k \nabla T) = - h \left( T - T_\infty \right) .

Here :math:`h` is the film coefficient in watts per square metre per kelvin and
:math:`T_\infty` the temperature of the surrounding fluid far from the surface.
The minus sign is what makes the condition physical under the library's
convention: where the surface is hotter than the fluid the bracket is positive,
the entering flux is negative, and heat leaves.  Only the temperature
*difference* enters, so any consistent scale works, which is not true of
radiation below.  The object is ``convective_heat_flux_boundary_condition``; it is nonlinear in
nothing but :math:`T` itself and is differentiated exactly, so a linear
conduction problem carrying it still converges in a single Newton step.  Its
default ambient temperature is zero, which cools the surface towards zero and is
rarely intended when the variable is in degrees Celsius.

The **grey-body radiation** condition applies the Stefan-Boltzmann law to a
surface exchanging radiation with large surroundings:

.. math::
   :label: radiation

   \mathbf{n} \cdot (k \nabla T)
   = - \varepsilon \sigma \left( T^4 - T_\infty^4 \right) ,

where :math:`\varepsilon` is the total hemispherical emissivity, dimensionless
and between zero and one, and
:math:`\sigma = 5.670374419 \times 10^{-8}\ \mathrm{W/m^2/K^4}` is the
Stefan-Boltzmann constant.  The two are multiplied together once when the object
is constructed, so neither can be varied afterwards.  Two warnings belong here.
The fourth powers demand an **absolute temperature scale**: a problem posed in
degrees Celsius produces a silently wrong answer rather than an error, because
nothing in the code can detect the mistake.  And :eq:`radiation` is strongly
nonlinear, so radiation overwhelms conduction at high temperature and a poor
initial guess can prevent convergence.  Under Newton's method the fourth power
is differentiated exactly; under direct iteration the code instead factors the
difference as
:math:`T^4 - T_\infty^4 = ( T^2 + T_\infty^2 )( T + T_\infty )( T - T_\infty )`
and evaluates the first two brackets at the previous iterate, leaving a term
linear in the current temperature.  The object is ``radiative_heat_flux_boundary_condition``, and
its default ambient temperature of zero models radiation into deep space.

Temperature-dependent conductivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most materials conduct differently when hot.  ``heat_conduction`` expresses this
by multiplying a base conductivity, itself allowed to vary with position and
time, by a polynomial in the temperature:

.. math::
   :label: kpoly

   k(\mathbf{x}, t, T) = k_0(\mathbf{x}, t)
   \left( c_0 + c_1 T + c_2 T^2 + \cdots \right) .

The base value :math:`k_0` is the ``thermal_conductivity`` parameter, or a
material property when ``thermal_conductivity_property`` names one; naming a
property replaces :math:`k_0` alone, and the polynomial still multiplies it.
The coefficients are the ``temperature_polynomial`` list, evaluated by Horner's
rule from the highest downwards.  Two details of :eq:`kpoly` have caught users
out.  The polynomial **multiplies** the base value rather than replacing it, so
a list supplied by the user must carry its own constant term: the default list
:math:`\{1\}` leaves the conductivity untouched, and a material with
:math:`k = 20\,(1 + 0.01\,T)` is written as a base value of :math:`20` together
with the list :math:`\{1, 0.01\}`.  Further, the polynomial is written in the
same temperature units as the variable and not relative to a reference
temperature, so its coefficients must be converted along with the rest of the
problem if the scale changes.

For the solver the consequence is that :eq:`energy` is no longer linear in
:math:`T`, the flux now containing the product of a function of :math:`T` with
:math:`\nabla T`.  One linear solve no longer suffices, and which iteration
replaces it decides where the temperature inside :eq:`kpoly` is taken from.
Under Newton's method the polynomial sees the current iterate as a
differentiated quantity, so the Jacobian gains the term
:math:`(\partial k / \partial T) \nabla T` alongside the usual
:math:`k \nabla \psi`, and convergence is quadratic.  Under direct iteration the
polynomial sees the previous iterate as a constant, the matrix then resembles
that of a linear problem with a spatially varying conductivity, and convergence
is linear.  One accessor expresses the distinction: the conductivity is built
from ``coefficientValue``, which returns the current iterate in Newton mode and
the lagged one in Picard mode, while the gradient multiplying it is always the
current differentiated one.

Verification of the heat transfer module
----------------------------------------

The module is checked against the published dual mesh results of Reddy's book
rather than against itself, in ``tests/python/test_heat_transfer.py``.  The
cooling fin of Example 5.3.1, a one-dimensional problem with a Robin condition
at the tip, is reproduced on ten and twenty elements against the nodal
temperatures and base heat flow of Table 5.3.1, the temperatures agreeing within
:math:`6\times10^{-3}\ \mathrm{K}` and the recovered reaction of
:math:`4807\ \mathrm{W}` within three parts in ten thousand.  The bus bar of
Example 5.4.3, which combines volumetric generation, two fixed edges and a
convecting top surface, reproduces every printed digit of Table 5.4.3 on a
:math:`10 \times 5` mesh; that table is captioned as :math:`20 \times 10`, but
the accompanying figure shows the coarser mesh and its values agree with those
quoted for :math:`10 \times 5` elsewhere in the book, so the test uses the mesh
the numbers came from.  The nonlinear plate of Example 6.3.1, with
:math:`k = k_0 (1 + 100\,T)`, matches the dual mesh column of Table 6.3.1 to
within three hundredths of a degree by Newton's method and by direct iteration
alike, that tolerance being set by the book's own convergence criterion of
:math:`10^{-3}` relative rather than by the discretization.

Viscous incompressible flow
---------------------------

The governing equations
^^^^^^^^^^^^^^^^^^^^^^^

An isothermal flow of an incompressible Newtonian fluid is governed by
conservation of mass and of momentum.  With :math:`\mathbf{v}` the velocity,
:math:`P` the pressure, :math:`\rho` the constant density, :math:`\mu` the
dynamic viscosity in pascal seconds and :math:`\mathbf{f}` a body force per unit
volume, these are

.. math::
   :label: continuity

   \nabla \cdot \mathbf{v} = 0

and

.. math::
   :label: momentum

   \rho \left( \frac{\partial \mathbf{v}}{\partial t}
   + \left( \mathbf{v} \cdot \nabla \right) \mathbf{v} \right)
   = - \nabla P
   + \nabla \cdot \left[ \mu \left( \nabla \mathbf{v}
     + \nabla \mathbf{v}^{\mathsf{T}} \right) \right]
   + \mathbf{f} .

Equation :eq:`continuity` states that the volume of a material element does not
change.  In :eq:`momentum` the left-hand side is mass times acceleration, the
acceleration being the material derivative: the local rate of change plus the
convective term :math:`(\mathbf{v} \cdot \nabla) \mathbf{v}`, which accounts for
a particle being carried into a region where the velocity differs.  On the right
stands the divergence of the Cauchy stress, split into the pressure and the
viscous stress :math:`\mu (\nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}})`
of a Newtonian fluid, together with the body force.  The module treats steady
flow, so :math:`\partial \mathbf{v} / \partial t` is dropped; a transient flow
would add the generic ``time_derivative`` kernel to each momentum equation.  The
ratio of the convective to the viscous term defines the Reynolds number
:math:`Re = \rho V L / \mu` for a problem of characteristic speed :math:`V` and
length :math:`L`.

Why the pressure is the difficulty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conduction gives one equation for one field, and that field appears in it.  The
Navier-Stokes system lacks this structure.  In two dimensions there are three
fields, :math:`u`, :math:`v` and :math:`P`, and three equations, two momentum
balances and continuity, but the pressure appears in the momentum equations and
nowhere in :eq:`continuity`.  No equation has the pressure as its principal
unknown, and no constitutive law delivers it from the kinematics.  The pressure
is not a thermodynamic state variable here: it is the Lagrange multiplier
enforcing the constraint :eq:`continuity`, taking whatever value keeps the flow
divergence free, and in a fully enclosed flow it is fixed only to within an
additive constant.

The consequence for a method that interpolates velocity and pressure together is
sharp.  The two interpolation spaces cannot be chosen independently; they must
jointly satisfy the Ladyzhenskaya-Babuska-Brezzi, or inf-sup, condition, which
requires the pressure space to be poor enough relative to the velocity space
that every discrete pressure is felt by some discrete velocity.  A pair
violating it yields a singular or nearly singular system and pressure fields
polluted by spurious modes, the checkerboard oscillation being the familiar one.
Equal-order interpolation of velocity and pressure, which is what a library
built on a single family of Lagrange elements would otherwise reach for, is
precisely the choice that fails.

The penalty formulation
^^^^^^^^^^^^^^^^^^^^^^^

The library avoids the mixed problem altogether by the penalty formulation of
Chapter 9 of Reddy's book.  Rather than treating the constraint exactly with a
multiplier, the fluid is treated as very slightly compressible, through the
constitutive assumption

.. math::
   :label: penalty

   P = - \gamma \, \nabla \cdot \mathbf{v} ,

in which :math:`\gamma` is the penalty parameter, a number with the units of
viscosity.  The sign is fixed by physics: a fluid element being compressed has
:math:`\nabla \cdot \mathbf{v} < 0` and must be at raised pressure.  As
:math:`\gamma \to \infty` a bounded pressure forces
:math:`\nabla \cdot \mathbf{v} \to 0` and the incompressible problem returns; at
finite :math:`\gamma` the flow carries a residual dilatation of order
:math:`P / \gamma`.  The benefit is that :eq:`penalty` removes the pressure as
an unknown.  Substituting it into :eq:`momentum` expresses :math:`P` through the
velocity gradients and leaves two equations in the two velocity components, both
of the same second-order form as the conduction equation and hence expressible
in the canonical form :eq:`canonical_hf`.  Continuity is no longer imposed
separately; the penalty term enforces it approximately, and nothing else in the
framework has to know that a constraint is present.

The choice of :math:`\gamma` is a compromise and is the parameter most likely to
be set badly.  It must be measured against the viscosity, since the penalty term
competes with the viscous term within the same equation, and a ratio of roughly
:math:`10^4` to :math:`10^7` works, that is, :math:`\gamma \approx 10^4 \mu` to
:math:`10^7 \mu`.  The default :math:`\gamma = 10^8` suits a viscosity of order
unity.  At the low extreme the constraint is weakly enforced, the computed flow
is measurably compressible and the answer belongs to a different physical
problem.  At the high extreme the penalty contribution dominates every matrix
entry, the viscous contribution is lost to round-off against it, and the system
becomes so ill-conditioned that even a direct factorization loses accuracy, so
the velocity field degrades rather than improving.  A useful check is the
residual divergence, which should behave as
:math:`|\nabla \cdot \mathbf{v}| \sim P / \gamma` and which the tests confirm
stays below :math:`10^{-5}` for the cavity at the default setting.

Reduced integration and locking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The penalty term must be integrated with a **reduced** quadrature rule, and this
is not a refinement but the thing that makes the formulation work at all.  The
argument is one of counting.  Integrating the penalty term with a rule of
:math:`n` points per element imposes, in the limit of large :math:`\gamma`, the
discrete constraint :math:`\nabla \cdot \mathbf{v} = 0` at each of those points,
so a mesh of :math:`N_e` elements carries :math:`n N_e` constraints.  Against
this stands the number of velocity degrees of freedom: a structured mesh of
four-node quadrilaterals has about as many nodes as elements, hence about
:math:`2 N_e` velocity unknowns in two dimensions, fewer once the wall
conditions are applied.  With the full :math:`2 \times 2` Gauss rule,
:math:`n = 4` and there are some :math:`4 N_e` constraints against
:math:`2 N_e` unknowns.  A system with twice as many constraints as unknowns has
in general only the trivial solution, and that is what the computation returns: a
velocity field far too stiff and, in the limit, identically zero.  This failure
is locking, and refining the mesh does not cure it, because refinement
multiplies constraints and unknowns in the same proportion.

The one-point rule sets :math:`n = 1`, giving :math:`N_e` constraints against
about :math:`2 N_e` unknowns, a ratio of two unknowns per constraint.  That is
the ratio the continuum problem itself has, since at every point of a
two-dimensional flow two velocity components are subject to one incompressibility
condition, and it is the value at which the discrete problem neither locks nor
leaves the constraint unenforced.  ``penalty_incompressibility`` accordingly
defaults to the ``midpoint`` rule and to ``reduced_integration = true``, both
against the framework's own defaults, the latter meaning that the velocity
gradients are evaluated at the element centroid while the geometric integration
proceeds as usual.  This is selective reduced integration: the viscous term
keeps its full rule and only the constrained term is under-integrated.  A fuller
rule on the penalty term is available, and its only proper use is to demonstrate
the locking.

The same counting decides which discretisations can use the formulation.  The
vertex-centred finite volume method replaces the interpolated gradient at a
control domain interface by a two-point difference along the edge; applied to
the penalty term, that would impose the constraint once per edge rather than
once per element and lock the flow, so the replacement is not made where a
kernel asks for reduced integration.  With that exception the vertex-centred
method gives the same cavity flow as the dual mesh method.  The cell-centred
finite volume method has no element interior to integrate over: its fluxes are
formed at faces, from reconstructed cell gradients, so it imposes the
constraint at every face.  On a lid-driven cavity the computed velocities were
five orders of magnitude too small.  ``penalty_incompressibility`` therefore
refuses the cell-centred method; incompressible flow on that method needs a
pressure-velocity coupling, which is not implemented.

The equations as the code assembles them
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing :math:`u_i` for the :math:`i`-th velocity component and
:math:`\mathbf{e}_i` for the corresponding unit vector, the momentum equation
assembled by the module, one instance per component, is

.. math::
   :label: penalty_momentum

   \rho \left( \mathbf{v} \cdot \nabla \right) u_i
   - \nabla \cdot \left[ \mu \left( \nabla u_i
     + \left( \nabla \mathbf{v} \right)_i \right)
     + \gamma \left( \nabla \cdot \mathbf{v} \right) \mathbf{e}_i \right]
   - f_i = 0 ,

which is :eq:`canonical_hf` with contributions from three kernels and,
optionally, a body force.  Term by term:

* ``viscous_stress`` contributes the flux
  :math:`\mathbf{F} = \mu ( \nabla u_i + (\nabla \mathbf{v})_i )` and no source;
  its :math:`d`-th component is
  :math:`\mu ( \partial u_i / \partial x_d + \partial u_d / \partial x_i )`, so
  in two dimensions the :math:`u` equation carries
  :math:`(2\mu\,\partial u/\partial x,\ \mu(\partial u/\partial y +
  \partial v/\partial x))`.  The viscosity is evaluated from position and time
  only, which makes this kernel Newtonian; a shear-rate dependent viscosity
  needs a material and a kernel written for the purpose.
* ``penalty_incompressibility`` contributes the flux
  :math:`\mathbf{F} = \gamma (\nabla \cdot \mathbf{v}) \mathbf{e}_i`, that is
  :math:`\gamma (\partial u / \partial x + \partial v / \partial y +
  \partial w / \partial z)` in the direction of this instance's component and
  zero in the others, and no source.  This is the kernel carrying the reduced
  rule.
* ``convective_inertia`` contributes no flux and the source
  :math:`S = \rho (\mathbf{v} \cdot \nabla) u_i`.  The transporting velocity
  :math:`\mathbf{v}` is taken through the coefficient accessor, so it is the
  current iterate under Newton's method and the previous one under direct
  iteration, while the gradient :math:`\nabla u_i` it multiplies is always the
  current differentiated one.  A Stokes flow is obtained by omitting this kernel
  rather than by setting the density to zero.
* ``body_force``, if present, contributes no flux and the source
  :math:`S = -f_i`, the sign again being what makes a positive intensity drive
  the velocity up.

Each object takes the same ``velocities`` list, in coordinate order, and a
zero-based ``component`` index into it.  The index is range checked, but its
agreement with the ``variable`` parameter is not, so a mismatch silently
assembles one momentum equation into another; the helper
``dualmesh.physics.add_incompressible_flow`` exists partly to remove that
opportunity, adding the viscous and penalty kernels for every component, the
inertia kernel when a non-zero density is given, and the pressure material
described next.

The natural boundary quantity of :eq:`penalty_momentum` follows the same rule as
in conduction:

.. math::

   \mathbf{n} \cdot \mathbf{F}
   = \sum_d n_d \, \mu \left( \frac{\partial u_i}{\partial x_d}
     + \frac{\partial u_d}{\partial x_i} \right) - P \, n_i ,

where :eq:`penalty` was used to replace :math:`\gamma \nabla \cdot \mathbf{v}`
by :math:`-P`.  This is the :math:`i`-th component of the surface traction, so a
boundary with no condition at all is traction free, which is the usual outflow
condition.  Walls and moving lids are imposed instead as ``Dirichlet_boundary_condition``
conditions on the velocity components.

Recovering the pressure
^^^^^^^^^^^^^^^^^^^^^^^

The pressure was eliminated, not discarded, and :eq:`penalty` recovers it from
the converged velocity field.  The ``penalty_pressure`` material declares the
property ``pressure`` and evaluates :math:`P = -\gamma \nabla \cdot \mathbf{v}`
wherever it is asked for.  Two cautions apply.  The value of :math:`\gamma`
given to the material must be identical to the one given to
``penalty_incompressibility``: the two objects hold separate copies, nothing
checks that they agree, and a mismatch scales the recovered pressure by the
ratio of the two, giving a plausible looking field of the wrong magnitude.  And
the pressure should be evaluated at the element centroids, which are the
reduced-order points at which the constraint was imposed; the divergence there
is the quantity the formulation controlled, and the recovered pressure is
markedly more accurate there than at the nodes.

Solving the flow equations
--------------------------

For a Stokes flow the assembled system is linear and one solve suffices.  Once
``convective_inertia`` is added the system becomes nonlinear, because the source
:math:`\rho (\mathbf{v} \cdot \nabla) u_i` is quadratic in the unknowns: the
velocity transports itself.  The library offers two ways to handle this, and the
tests exercise both on the same problem so the answers can be compared.

Newton's method with load stepping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default solver linearizes the residual exactly.  Writing
:math:`\mathbf{R}(\mathbf{U})` for the assembled residual vector, each iteration
solves

.. math::

   \mathbf{J}(\mathbf{U}^r) \, \delta \mathbf{U} = -\mathbf{R}(\mathbf{U}^r) ,
   \qquad \mathbf{U}^{r+1} = \mathbf{U}^r + \delta \mathbf{U} ,

with :math:`\mathbf{J} = \partial \mathbf{R} / \partial \mathbf{U}` obtained by
forward automatic differentiation of the same kernel code that computes the
residual.  The Jacobian is therefore exact to machine precision and contains
both the :math:`\rho (\delta \mathbf{v} \cdot \nabla) u_i` and the
:math:`\rho (\mathbf{v} \cdot \nabla) \delta u_i` parts of the convective term,
so convergence is quadratic near the solution.  Iteration stops when the
residual norm falls below an absolute tolerance, or below a relative multiple of
its initial value, or when the relative change in the solution falls below the
step tolerance.

What goes wrong as the Reynolds number rises is that the domain of quadratic
convergence shrinks.  At :math:`Re = 1000` in the lid-driven cavity the
recirculating flow is far from the rest state, and an iteration started from
zero velocity takes a first step so large that it lands outside the basin of
attraction and diverges.  The remedy is **load stepping**: the boundary velocity
is applied in a sequence of load factors, each step solved to convergence and
its solution used as the initial guess for the next.  The cavity test uses the
factors :math:`0.1, 0.25, 0.5, 0.75, 1.0` and marks the lid condition with
``scale_with_load`` so that it is the quantity ramped; each intermediate problem
is then close enough to its predecessor for Newton's method to succeed, and the
computation traces a path through a family of flows of increasing Reynolds
number instead of attempting the final one in a single leap.

Direct iteration with relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The alternative is direct, or Picard, iteration, which is the linearization used
in the book.  Every nonlinear coefficient is evaluated at the previous iterate
instead of the current one, so the transporting velocity in
:math:`\rho (\mathbf{v} \cdot \nabla) u_i` is frozen and each iteration is a
linear convection-diffusion solve.  The matrix is cheaper to form and the method
is far less sensitive to the starting point, but convergence is linear rather
than quadratic, and at higher Reynolds number the iteration tends to overshoot
and oscillate between two states without settling.  The cure is
under-relaxation: after the linear solve produces a tentative iterate
:math:`\tilde{\mathbf{U}}^{r+1}`, the code blends it with the previous one,

.. math::
   :label: relaxation

   \mathbf{U}^{r+1} = \left( 1 - \beta \right) \tilde{\mathbf{U}}^{r+1}
   + \beta \, \mathbf{U}^{r} ,

where :math:`\beta` is the ``relaxation`` parameter, written :math:`\gamma` in
the book's Eq. (6.2.15) and not to be confused with the penalty parameter of
:eq:`penalty`.  Read the formula carefully, because the weight sits on the
**old** iterate: :math:`\beta = 0`, the default, is the unrelaxed method, a value
near one barely moves at all, and the useful range for a stubborn flow is around
one half.  Relaxation is skipped on the first iteration of each load step, so
the method takes one full step before it begins damping.  The same mechanism
serves both nonlinear solvers, though direct iteration is what ordinarily needs
it.

The two strategies meet the same failure from opposite directions.  Newton's
method keeps its fast local convergence and manages the poor starting point by
continuation in the loading, at the cost of solving a sequence of problems.
Direct iteration accepts slow convergence in exchange for robustness and manages
the oscillation by damping each step, at the cost of many more iterations: the
cavity at :math:`Re = 1000` is allowed sixty Newton iterations spread over five
load steps, against four hundred relaxed direct iterations.  Both reach the same
flow.

The pressure-velocity formulation
---------------------------------

The penalty method removes the pressure from the unknowns, which is economical,
but it has three costs.  The penalty parameter :math:`\gamma` must be large
for the divergence to be small, which makes the matrix badly conditioned; the
penalty term must be integrated with a reduced rule, which has no counterpart
in the cell-centred finite volume method; and the pressure is only recovered
afterwards, element by element.  The *pressure-velocity* (or *mixed*)
formulation, selected with ``formulation="pressure"`` in
``dualmesh.physics.add_incompressible_flow``, keeps the pressure :math:`p` as
an unknown field and solves the mass equation alongside the momentum
equations.  It works with all four methods.

The equations
^^^^^^^^^^^^^

The momentum equation of component :math:`i` has as its flux the :math:`i`-th
row of the Cauchy stress,

.. math::
   :label: pv_momentum

   \mathbf{F}_i = \mu \left( \nabla u_i + \frac{\partial \mathbf{v}}{\partial x_i} \right)
   - p\, \mathbf{e}_i ,
   \qquad
   S_i = \rho \left( \frac{\partial u_i}{\partial t} + \mathbf{v} \cdot \nabla u_i \right) - f_i ,

and the mass equation :math:`\nabla \cdot \mathbf{v} = 0` is written in the
canonical form :math:`-\nabla \cdot \mathbf{F} + S = 0` with the flux
:math:`\mathbf{F} = -\mathbf{v}` and no source.  ``viscous_stress`` supplies
the viscous part of :eq:`pv_momentum`, the new kernel ``pressure_gradient`` the
pressure part :math:`-p\,\mathbf{e}_i`, and ``mass_conservation`` the mass
equation, which is attached to the pressure variable.  Because the pressure
sits inside the flux, the natural boundary quantity of a momentum equation is
the full traction :math:`(\boldsymbol{\sigma} \mathbf{n})_i`, and a boundary
without a velocity condition is a traction-free outlet.  In axisymmetric
coordinates without swirl the radial equation also carries the hoop stress:
the source gains :math:`-\sigma_{\theta\theta}/r = -(2 \mu u_r / r - p)/r`,
which ``viscous_stress`` and ``pressure_gradient`` add between them.

Why equal-order interpolation needs stabilisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pressure in an incompressible flow is not governed by an equation of its
own.  It is the Lagrange multiplier of the constraint
:math:`\nabla \cdot \mathbf{v} = 0`, and a discretisation determines it only
if the discrete velocity space is rich enough to "see" every discrete
pressure mode.  This is the inf-sup, or Ladyzhenskaya-Babuška-Brezzi,
condition.  Interpolating velocity and pressure with the same functions
violates it: on a quadrilateral mesh the checkerboard pressure, alternating
:math:`+1` and :math:`-1` from node to node, produces no force on any discrete
velocity and is therefore invisible to the equations.  There are two cures.
One is to enrich the velocity, as the Taylor-Hood element does with quadratic
velocity and linear pressure.  The other, used here, is to *stabilise*: to add
to the mass equation a term that controls the pressure gradient and vanishes
for the exact solution.

The stabilised mass flux
^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mathbf{r}_m` be the residual of the momentum equation in its
strong form, without the viscous term,

.. math::
   :label: pv_residual

   \mathbf{r}_m = \rho \left( \frac{\partial \mathbf{v}}{\partial t}
   + (\mathbf{v} \cdot \nabla) \mathbf{v} \right) + \nabla p - \mathbf{f} .

``mass_conservation`` replaces the mass flux :math:`-\mathbf{v}` by

.. math::
   :label: pv_mass_flux

   \mathbf{F} = \tau\, \mathbf{r}_m - \mathbf{v} .

The exact solution makes :math:`\mathbf{r}_m` equal to the viscous force
:math:`\nabla \cdot (\mu \nabla \mathbf{v})`, which is zero for a linear
velocity field, so the added term is (almost) consistent.  For the Galerkin
finite element method, integrating :math:`\nabla \psi \cdot \mathbf{F}` over
the domain gives the mass equation plus
:math:`\int \tau \nabla \psi \cdot \mathbf{r}_m\, d\Omega`, which is exactly
the pressure-stabilising Petrov-Galerkin (PSPG) method of
[HughesFrancaBalestra1986]_; its pressure-gradient part,
:math:`\int \tau \nabla \psi \cdot \nabla p\, d\Omega`, is a small pressure
Laplacian that removes the checkerboard mode.  For the control volume methods
the same flux is integrated over the faces of each control volume, and the
face mass flux becomes the interpolated velocity corrected by
:math:`\tau (\nabla p - \dots)`.  This is the idea behind the momentum
interpolation of [RhieChow1983]_, which cell-centred codes use to avoid
checkerboard pressures on colocated grids, written here with the whole momentum
residual rather than the pressure gradient alone.

The parameter :math:`\tau` has the units of a time divided by a density, so
that :math:`\tau \mathbf{r}_m` is a velocity.  It is the one of
[Tezduyar1991]_ (see also [TezduyarMittalRayShih1992]_), written per unit
density so that it also covers the Stokes equations:

.. math::
   :label: pv_tau

   \tau = \left[ \left( \frac{2 \rho}{\Delta t} \right)^2
   + \left( \frac{2 \rho |\mathbf{v}|}{h} \right)^2
   + \left( \frac{4 \mu}{h^2} \right)^2 \right]^{-1/2} .

The three terms are the reciprocal time scales of the unsteadiness, of
convection across an element and of diffusion across it; the first is absent
in a steady solve.  The element size is :math:`h = (c V_e)^{1/d}`, with
:math:`V_e` the element volume and :math:`c = 1` for quadrilaterals and
hexahedra, 2 for triangles and prisms and 6 for tetrahedra and pyramids, so
that every element of a structured grid of spacing :math:`h` gets :math:`h`;
it is halved for a quadratic element.  With inertia, ``momentum_stabilization``
adds the streamline-upwind term of [BrooksHughes1982]_ to each momentum
equation as the flux :math:`\rho \tau\, r_{m,i}\, \mathbf{v}`, which damps the
oscillations of convection-dominated flow; ``add_incompressible_flow`` adds it
whenever the density is non-zero.

Every force in the momentum equations must also appear in
:math:`\mathbf{r}_m`, because a force missing there makes the stabilisation
push the flow against it.  ``add_incompressible_flow`` passes its
``body_force`` and its Boussinesq ``buoyancy`` to the stabilisation, with the
same load scaling as the force itself.  Without the buoyancy in
:math:`\mathbf{r}_m` the natural convection benchmark fails at high Rayleigh
number, because there :math:`\tau \nabla p` is much larger than the velocity.

The omitted viscous term, :math:`-\nabla \cdot (\mu \nabla \mathbf{v})`, needs
second derivatives, which the kernels do not have.  It vanishes on linear
simplices and on bilinear elements that are parallelograms, and omitting it
keeps the optimal first order of the energy norm.  On quadratic elements it
does not vanish, the omission becomes an inconsistency, and the velocity
error stays at second order in :math:`L^2` instead of third.  The
Taylor-Hood element, which needs no stabilisation, is the quadratic choice.

The Taylor-Hood element
^^^^^^^^^^^^^^^^^^^^^^^

The other cure for the checkerboard is to make the velocity space richer than
the pressure space.  The element of [TaylorHood1973]_ interpolates the velocity
with the quadratic shape functions of the element and the pressure with the
linear shape functions of its corners: :math:`P_2`-:math:`P_1` on triangles
and tetrahedra, :math:`Q_2`-:math:`Q_1` on quadrilaterals and hexahedra.  The
pair satisfies the discrete inf-sup condition

.. math::
   :label: pv_inf_sup

   \beta_h = \inf_{q_h \in Q_h} \sup_{\mathbf{v}_h \in V_h}
   \frac{(q_h, \nabla \cdot \mathbf{v}_h)}
        {|\mathbf{v}_h|_1\, \|q_h\|_0} \;\ge\; \beta_0 > 0 ,

with :math:`\beta_0` independent of the mesh size [BoffiBrezziFortin2013]_, so
the Galerkin equations, with no stabilisation at all, determine a unique
pressure.  On a smooth solution the errors fall as

.. math::
   :label: pv_taylor_hood_rates

   \|\mathbf{v} - \mathbf{v}_h\|_0 = O(h^3), \qquad
   |\mathbf{v} - \mathbf{v}_h|_1 = O(h^2), \qquad
   \|p - p_h\|_0 = O(h^2) .

The mass equation is the unstabilised flux :math:`\mathbf{F} = -\mathbf{v}`,
tested with the linear pressure shape functions, and the momentum equations
are those above, tested with the quadratic ones.  Streamline stabilisation is
not offered with this element: its momentum residual :eq:`pv_residual` omits
the viscous term, which a quadratic velocity does not make vanish, so SUPG
would be inconsistent and would cost the third order of the velocity.

**How a first-order variable lives on a quadratic mesh.**  A variable declared
with ``order="first"`` is interpolated on every quadratic element with the
shape functions :math:`N^c_a` of the linear element that has the same corners
(``Tri3`` for ``Tri6``, ``Quad4`` for ``Quad9`` and ``Quad8``, ``Tet4`` for
``Tet10``, ``Hex8`` for ``Hex27`` and ``Hex20``).  They are evaluated at the
same reference point as the quadratic ones, and their physical gradients use
the Jacobian :math:`\mathbf{J}` of the element's own quadratic map,

.. math::

   \nabla N^c_a = \mathbf{J}^{-T} \hat{\nabla} N^c_a ,

so on a curved element the pressure is linear in the reference coordinates,
which is the standard construction.  The corners are the first nodes of every
quadratic element, in the order of the linear element, and a unit test checks
that the corner basis is a partition of unity and reproduces a linear field
and its gradient exactly on an affinely mapped element of every type.

The global numbering keeps one slot per node and variable, so the numbering,
the parallel exchange and the block structure of the matrix (one block per
node, which PETSc's field splits use) are the same for every problem.  The
slots of a first-order variable at the mid-edge, mid-face and interior nodes
carry no equation: they are inactive, with an identity row in the Jacobian,
exactly like a variable outside its blocks.  After every change of the
solution they are set to the value of the linear field there.  Because the
quadratic shape functions reproduce every linear function of the reference
coordinates, interpolating those nodal values with the element's own shape
functions gives back exactly the linear pressure, so sampling, output, error
norms and every other post-processing routine see the right field without
knowing about the two orders.  Inside an element the local degrees of freedom
are numbered compactly (every variable at the corners, then only the
quadratic ones at the other nodes), and the automatic differentiation seeds
one derivative per local degree of freedom: 22 on a ``Quad9`` element instead
of 27, and :math:`27 \times 3 + 8 = 89` on a ``Hex27`` element.

**The linear systems.**  With :math:`\mathbf{A}` the momentum block and
:math:`\mathbf{B}` the discrete divergence, each Newton step solves

.. math::

   \begin{bmatrix} \mathbf{A} & \mathbf{B}^T \\ \mathbf{B} & \mathbf{0}
   \end{bmatrix}
   \begin{bmatrix} \delta \mathbf{v} \\ \delta \mathbf{p} \end{bmatrix}
   = - \begin{bmatrix} \mathbf{R}_v \\ \mathbf{R}_p \end{bmatrix} ,

a saddle point system whose pressure block is exactly zero.  An incomplete LU
factorisation meets a zero pivot on the first pressure row, so the default
linear solver (``linear_solver="automatic"``) factorises these systems
directly whatever their size.  In two dimensions that is fast; in three
dimensions its cost grows quickly (a ``Tet10`` mesh with about 20 000 unknowns
takes about 4 s per factorisation in one thread), and a large problem is
better solved with PETSc (``linear_solver="petsc"``), for instance with a
Schur complement field split, or with MUMPS in parallel.

Verification


``tests/python/test_taylor_hood.py`` checks the element in four ways.

*Orders of convergence*, by manufactured solutions, measured on the finest
pair of meshes (the theoretical values are 3, 2 and 2):

.. list-table:: Observed orders of the Taylor-Hood element
   :header-rows: 1
   :widths: 44 18 18 20

   * - Mesh and problem
     - :math:`\|\mathbf{v}-\mathbf{v}_h\|_0`
     - :math:`|\mathbf{v}-\mathbf{v}_h|_1`
     - :math:`\|p-p_h\|_0`
   * - ``Quad9``, distorted, Stokes and Navier-Stokes
     - 3.00
     - 2.00
     - 2.00
   * - ``Tri6``, Navier-Stokes
     - 3.00
     - 1.99
     - 2.4 (still approaching 2 from above)
   * - ``Quad8`` (serendipity velocity), Navier-Stokes
     - 3.00
     - 2.00
     - 2.00
   * - ``Quad9`` and ``Tri6``, axisymmetric, including the axis
     - 3.00
     - 2.00
     - 2.00 to 2.07
   * - ``Quad9``, enclosed cavity, pressure pinned at a corner
     - 3.00
     - 2.00
     - 2.7 or better
   * - ``Tet10``, three dimensions (:math:`n = 2, 4`)
     - 2.96
     - 1.97
     - 2.25
   * - ``Hex27``, three dimensions (:math:`n = 2, 4, 6`)
     - 3.00
     - 2.00
     - 2.02

The enclosed case includes a velocity that crosses the boundary, for which the
stabilised equal-order element produces an :math:`O(1)` pressure spike at the
pin (see below); the Taylor-Hood element, which has no pressure Laplacian to
carry the imbalance, shows none, and its pressure still converges at second
order.

*The inf-sup condition itself*, by the numerical test of
[ChapelleBathe1993]_.  With :math:`\mathbf{A}` the vector Laplacian (whose
energy is :math:`|\mathbf{v}_h|_1^2`), :math:`\mathbf{B}` the discrete
divergence and :math:`\mathbf{M}` the pressure mass matrix, for velocities
that vanish on the boundary, :math:`eta_h^2` is the smallest non-zero
eigenvalue :math:`\lambda` of

.. math::

   \mathbf{B} \mathbf{A}^{-1} \mathbf{B}^T \mathbf{q} = \lambda\, \mathbf{M} \mathbf{q}

(the smallest eigenvalue is zero, for the constant pressure that an enclosed
flow does not determine).  On the unit square:

.. list-table:: Discrete inf-sup constant :math:`eta_h` on an :math:`n 	imes n` mesh
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Pair
     - :math:`n = 4`
     - :math:`n = 8`
     - :math:`n = 12`
     - :math:`n = 16`
   * - :math:`Q_2`-:math:`Q_1` (Taylor-Hood)
     - 0.475
     - 0.463
     - 0.458
     - 0.455
   * - :math:`P_2`-:math:`P_1` (Taylor-Hood)
     - 0.368
     - 0.366
     - 0.366
     - 0.366
   * - :math:`Q_1`-:math:`Q_1`, no stabilisation
     - 0
     - 0
     - 0
     - 0
   * - :math:`Q_2`-:math:`Q_2`, no stabilisation
     - 0
     - 0
     - 0
     - 0

The Taylor-Hood constants settle to a mesh-independent value, while the
equal-order pairs have spurious pressure modes (their second eigenvalue is
zero to round-off) at every size.  The triangles use the structured "right"
diagonal, which leaves two corner triangles with every vertex on the boundary;
the constant is bounded all the same.  The same computation checks that the
pressure coupling of the momentum equations is the negative transpose of the
divergence to round-off, as the Galerkin method requires.

*Newton's method* converges quadratically (the exact Jacobian includes the
mixed-order assembly), and the *lid-driven cavity* at Re = 100 on
:math:`16 	imes 16` ``Quad9`` elements agrees with [Ghia1982]_ on both
centrelines within 0.01.

*Misuse is refused* with a message that says what to do: a first-order
variable with any method other than ``fem``; ``formulation="taylor_hood"`` on
a linear mesh or with another method; stabilisation or SUPG requested with
it; an existing pressure of the wrong order; and ``mass_conservation`` without
stabilisation on an equal-order pressure, which would otherwise return the
spurious modes.

Boundary conditions for the pressure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mass equation needs to know the flow through the boundary.
``mass_flux_boundary_condition`` supplies it as the natural boundary quantity
:math:`q = -\mathbf{v} \cdot \mathbf{n}`, computed from the discrete velocity;
``add_incompressible_flow`` places it on every side set.  For the finite
element method this reproduces the boundary term of PSPG exactly.  Summing the
mass equations of all control volumes gives
:math:`\oint \mathbf{v}_h \cdot \mathbf{n}\, ds = 0`, so the discrete flow
conserves mass globally, whatever the stabilisation.

An enclosed flow, with a velocity condition on every boundary, determines the
pressure only up to a constant.  ``pin_pressure`` fixes it with a
``point_Dirichlet_boundary_condition``, which replaces the mass equation of the entity nearest
to a point.  That equation is then no longer imposed, and the only place where
a net boundary flux :math:`\oint \mathbf{v}_h \cdot \mathbf{n}\, ds \neq 0`
can go.  With walls and a sliding lid the discrete net flux is zero and the
pin is harmless.  With prescribed velocities that cross the boundary,
however, the nodal interpolant of the data carries a net flux of order
:math:`h^2`, and the pressure Laplacian of the stabilisation, whose coefficient
is :math:`\tau \sim h^2`, turns it into a pressure spike of order one at the
pin.  Such a flow should have an outlet: a boundary with a traction condition
(or none) determines the pressure and needs no pin.

Verification of the pressure-velocity formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tests/python/test_pressure_velocity.py`` checks the formulation with the
manufactured solutions of ``dualmesh.mms.IncompressibleFlow``, whose exact
velocity is divergence free, whose right boundary is an outlet with the exact
traction and whose pressure has no symmetry.  With the density
:math:`\rho = 1` and the viscosity :math:`\mu = 1`, the observed orders on the
finest pair of meshes (:math:`32^2` and :math:`64^2` elements; :math:`8^3` and
:math:`16^3` in three dimensions) are:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Mesh
     - :math:`\|\mathbf{v} - \mathbf{v}_h\|_{L^2}`
     - :math:`|\mathbf{v} - \mathbf{v}_h|_{H^1}`
     - :math:`\|p - p_h\|_{L^2}`
   * - Distorted Quad4, node methods
     - 1.98 to 1.99
     - 1.01
     - 1.60 to 1.65
   * - Distorted Quad4, ``zfvm``
     - 1.99
     - 1.01
     - 1.58
   * - Distorted Tri3, node methods
     - 1.99
     - 1.00 to 1.01
     - 1.59 to 1.60
   * - Distorted Tri3, ``zfvm``
     - 1.99
     - 1.01
     - 1.26
   * - Axisymmetric Quad4 with the axis, all methods
     - 1.95 to 1.97
     - 1.03
     - 1.61 to 1.68
   * - Hex8, all methods
     - 1.81 to 1.83
     - 1.15 to 1.21
     - 1.59 to 1.66

The theory of the stabilised methods guarantees first order for the velocity
in :math:`H^1` and for the pressure in :math:`L^2`
[HughesFrancaBalestra1986]_, and second order for the velocity in :math:`L^2`;
all three are observed or exceeded.  The pressure is not controlled in
:math:`H^1`, and its :math:`H^1` error converges only at about half an order,
which is the behaviour usually reported for equal-order linear elements
[BrezziPitkaranta1984]_.

The physics is checked against two benchmarks.  For the lid-driven cavity at
:math:`Re = 100` on a :math:`32 \times 32` mesh, every method reproduces the
centreline velocities of [Ghia1982]_ to within 0.015, and to within 0.0074 on
:math:`64 \times 64`; at :math:`Re = 1000` on :math:`64 \times 64` the finite
element method gives the minimum horizontal velocity :math:`-0.381`, against
:math:`-0.383` in the reference.  The node-based methods need the two corner
nodes of the lid to belong to the walls: with the lid velocity there, the lid
"leaks" through the first row of elements and the vortex is noticeably too
weak at :math:`Re = 1000`.  For natural convection, with the buoyancy in the
stabilisation, the Nusselt number and velocity maxima on the
:math:`32 \times 32` graded mesh are within 2 % of [DeVahlDavis1983]_ for the
node methods up to :math:`Ra = 10^6`, and within 3.6 % for the cell-centred
method.  Newton's method converges quadratically for all four methods.

Verification of the fluids module
---------------------------------

The module is checked in ``tests/python/test_fluids.py`` against the penalty
results of Chapter 9 of Reddy's book.  The creeping flow squeezed between two
approaching plates, Example 9.8.3, is computed on the graded
:math:`20 \times 16` mesh and reproduces the horizontal velocity profiles of the
dual mesh column of Table 9.8.2 at two stations to within
:math:`2 \times 10^{-3}`; the same test takes the recovered penalty pressure at
the element centroids and confirms that it follows Nadai's approximate solution
:math:`P = 3 \mu V_0 (a^2 + y^2 - x^2) / (2 b^3)` in the interior, with a
correlation above :math:`0.999` and a magnitude correct to within five per cent,
the comparison being restricted to the interior because the corners of this
problem are singular.  The lid-driven cavity of Example 9.8.4 is computed on the
:math:`16 \times 20` mesh graded towards the lid and reproduces the centreline
profile of Table 9.8.3 to within :math:`2 \times 10^{-3}` at :math:`Re = 0`,
where the flow is Stokes, and again at :math:`Re = 1000`, where the same profile
is obtained twice over: once by Newton's method with the lid ramped in five load
steps, and once by relaxed direct iteration with :math:`\beta = 0.5`.  A further
test at :math:`Re = 400` confirms that the two nonlinear solvers agree with each
other to :math:`10^{-6}` in every nodal velocity, a stronger statement than
either comparison with the book, and a last one confirms that the divergence of
the computed field stays below :math:`10^{-5}` everywhere, as :eq:`penalty`
requires at the penalty parameter used.

Coupled heat transfer and flow: natural convection
--------------------------------------------------

When the density of a fluid depends on its temperature, the temperature drives
the flow and the flow carries the heat.  The **Boussinesq approximation**
[DeVahlDavis1983]_ keeps the density constant, :math:`\rho_0`, everywhere
except in the gravity term, where

.. math::

   \rho = \rho_0 \left[ 1 - \beta (T - T_0) \right]

with :math:`\beta` the volumetric expansion coefficient and :math:`T_0` a
reference temperature.  The constant part :math:`\rho_0 \mathbf{g}` of the
gravity force is balanced by a hydrostatic pressure, which leaves the body force

.. math::
   :label: boussinesq

   \mathbf{f} = -\rho_0 \beta (T - T_0) \, \mathbf{g} ,

so fluid warmer than :math:`T_0` rises.  In the canonical form of the momentum
equation of component :math:`i` the source is :math:`S = -f_i`, and this is
what ``Boussinesq_buoyancy`` adds.  The energy equation gains the advective term

.. math::
   :label: heatconvection

   \rho c_p \, \mathbf{v} \cdot \nabla T ,

added as a source by ``heat_convection``, in which :math:`\mathbf{v}` is the
velocity being solved for.  The non-conservative form is exact for an
incompressible flow, because
:math:`\nabla \cdot (\mathbf{v} T) = \mathbf{v} \cdot \nabla T + T \nabla
\cdot \mathbf{v}` and the last term vanishes.

The two terms couple the velocity and the temperature in both directions, and
the Jacobian of the coupled system has the blocks
:math:`\partial R_{\mathbf{v}} / \partial T` from :eq:`boussinesq` and
:math:`\partial R_T / \partial \mathbf{v}` from :eq:`heatconvection`.  Both
are computed by the automatic differentiation like every other block, so
Newton's method converges quadratically on the coupled problem.

**Verification.**  The benchmark is the square cavity of de Vahl Davis
[DeVahlDavis1983]_: side :math:`L`, the left wall at :math:`T = +1/2` and the
right at :math:`-1/2` (in units of the temperature difference), adiabatic top
and bottom, no slip everywhere, air with :math:`Pr = 0.71`, and the Rayleigh
number :math:`Ra = g \beta \Delta T L^3 / (\nu \kappa)` from :math:`10^3` to
:math:`10^6`.  With lengths scaled by :math:`L` and velocities by
:math:`\kappa / L`, the viscosity becomes :math:`Pr`, the conductivity and the
heat capacity one, and :math:`\rho_0 \beta g` becomes :math:`Ra \, Pr`.  The
average Nusselt number is the heat flow through the hot wall, which is the sum
of the reactions of the temperature equation there.  On a :math:`32 \times
32` mesh whose lines cluster at the walls (a cosine spacing), with the Rayleigh
number reached by continuation, the dual mesh method gives:

.. table:: Natural convection in a square cavity, dual mesh method, 32 by 32

   ============  ==========  ========  ===========  ========  ===========  ========
   :math:`Ra`    :math:`Nu`  ref.      :math:`u_m`  ref.      :math:`v_m`  ref.
   ============  ==========  ========  ===========  ========  ===========  ========
   :math:`10^3`  1.117       1.118     3.660        3.649     3.707        3.697
   :math:`10^4`  2.243       2.243     16.24        16.18     19.71        19.62
   :math:`10^5`  4.517       4.519     34.79        34.73     68.78        68.59
   :math:`10^6`  8.814       8.800     64.97        64.63     223.4        219.4
   ============  ==========  ========  ===========  ========  ===========  ========

Here :math:`u_m` is the largest horizontal velocity on the vertical centreline
and :math:`v_m` the largest vertical velocity on the horizontal centreline.  The
Nusselt numbers agree to within 0.2 per cent and the velocity maxima to within
0.6 per cent, except :math:`v_m` at :math:`Ra = 10^6`, 1.8 per cent high, where
the wall jet is thinnest; on a :math:`64 \times 64` mesh it is 221.0.  The
finite element and vertex-centred finite volume methods give Nusselt numbers
within 0.2 per cent of the same values.  The heat entering at the hot wall
leaves at the cold wall to round-off, and on a :math:`16 \times 16` mesh at
:math:`Ra = 10^4` the relative Newton steps of the last load step fall as :math:`2.3 \times 10^{-3}`, :math:`2.5 \times 10^{-6}`,
:math:`5.1 \times 10^{-12}`, which is quadratic convergence.  The tests are in
``tests/python/test_multiphysics.py`` and the problem is
``examples/natural_convection.py``.
