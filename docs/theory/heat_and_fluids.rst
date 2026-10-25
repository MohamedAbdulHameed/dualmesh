Heat transfer and incompressible flow
=====================================

This chapter sets out the two continuum models implemented by the
``heat_transfer`` and ``fluids`` modules: conduction of heat in a solid, with a
conductivity that may depend on the temperature, and the slow flow of a viscous
incompressible fluid.  Both are developed from the underlying balance law,
because the library requires neither a weak form nor element matrices.  Its
input is the governing equation written in the canonical conservation form

.. math::
   :label: canonical_hf

   \mathcal{R}(u) \equiv -\nabla \cdot \mathbf{F}(u, \nabla u, \mathbf{x}, t)
   + S(u, \nabla u, \mathbf{x}, t) = 0 ,

in which :math:`\mathbf{F}` is the flux of the conserved quantity and :math:`S`
collects all terms outside the divergence.  A *kernel* supplies
:math:`\mathbf{F}`, or :math:`S`, or both, and every discretization operates on
that single statement: the dual mesh control domain method integrates it over the
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
volumetric generation rate.  The minus sign on the surface integral appears
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
source.  ``heat_source`` supplies no flux and the source :math:`S = -q'''`,
with the sign chosen so that a positive generation rate adds heat to the body.
``heat_conduction_time_derivative`` supplies no flux and the discrete capacity
term
:math:`S = \rho c_p (T - T_{\text{old}}) / \Delta t`, with
:math:`T_{\text{old}}` the temperature at the start of the step.  That kernel is
marked as a *time kernel*, with two consequences: it is omitted entirely from
a steady solve, and it is not multiplied by the time-integration
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
boundary condition prescribes.  The equality is read from right to left: because
:math:`\mathbf{n}` points outwards, :math:`\mathbf{n} \cdot \mathbf{q}` is heat
leaving, so :math:`q_n` is **the heat flux entering the body**, in watts per
square metre, counted positive inwards.  The convention holds throughout the
library, in this module and in the reactions recovered at constrained nodes.  It
is stated explicitly here because the opposite convention is at least as common
elsewhere.  A positive prescribed flux heats the body, and a
reported reaction of :math:`+4817\ \mathrm{W}` at a fixed boundary means that
much power flows inwards through it.

The convention also determines the behaviour of a boundary that carries no
condition at all: such a boundary is insulated.  This follows from the
discretization itself and requires no additional code.  The term :math:`q_n`
appears in the discrete equation of every boundary node, and if no object writes
a value into it, it contributes nothing, which is the statement :math:`q_n = 0`.
Zero heat flux defines an adiabatic surface, so a boundary left out of the input
is valid and represents a perfectly insulated surface.

Boundary conditions
^^^^^^^^^^^^^^^^^^^

Exactly one member of the pair :math:`(T, q_n)` is specified at every boundary
point, and the module offers four ways of doing so.

The **prescribed temperature**, or essential, condition sets
:math:`T = g(\mathbf{x}, t)` and is imposed by replacing the discrete equation
of each affected node with that statement, the displaced equation being retained
so that the secondary variable can be recovered afterwards as a reaction.  Use
it wherever the surface temperature is controlled or known.  The object is
``Dirichlet``.

The **prescribed flux**, or natural, condition sets
:math:`\mathbf{n} \cdot (k \nabla T) = q(\mathbf{x}, t)`, with :math:`q` the
heat entering per unit area.  Use it for a surface heated by a known source such
as an electrical element, and use its default value of zero, or no condition at
all, for an insulated or symmetry surface.  The physics level writes it as
text with the heat *leaving* the body, ``"heat_flux = -q"``, and the object
level is ``Neumann``, with the flux entering the body as above.

The **convection** condition applies Newton's law of cooling, which models the
thermal resistance of the fluid boundary layer adjacent to the surface by one
lumped coefficient:

.. math::
   :label: convection

   \mathbf{n} \cdot (k \nabla T) = - h \left( T - T_\infty \right) .

Here :math:`h` is the film coefficient in watts per square metre per kelvin and
:math:`T_\infty` the temperature of the surrounding fluid far from the surface.
The minus sign makes the condition physically consistent with the library's
convention: where the surface is hotter than the fluid the bracket is positive,
the entering flux is negative, and heat leaves.  Only the temperature
*difference* enters, so any consistent temperature scale may be used, unlike the
radiation condition below.  The physics level writes it as text,
``"heat_flux = h*(temperature - T_inf)"``, with :math:`h` and
:math:`T_\infty` given as definitions, and the object level is ``Robin``
(``transfer_coeff`` and ``ambient_value``).  It depends on no unknown other
than :math:`T` itself and is differentiated exactly, so a linear conduction
problem carrying it still converges in a single Newton step.  The default
``ambient_value`` of ``Robin`` is zero, which cools the surface towards zero and
is rarely intended when the variable is in degrees Celsius.

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
is constructed, so neither can be varied afterwards.  Two cautions apply.
First, the fourth powers require an **absolute temperature scale**.  A problem
posed in degrees Celsius produces an incorrect answer without any error
message, because nothing in the code can detect the mistake.  Second,
:eq:`radiation` is strongly nonlinear, so radiation dominates conduction at high
temperature and a poor initial guess can prevent convergence.  Under Newton's
method the fourth power is differentiated exactly.  Under direct iteration the
code factors the difference as
:math:`T^4 - T_\infty^4 = ( T^2 + T_\infty^2 )( T + T_\infty )( T - T_\infty )`
and evaluates the first two brackets at the previous iterate, leaving a term
linear in the current temperature.  The object is
``radiative_heat_flux``, and its default ambient temperature
of zero models radiation into deep space.

Temperature-dependent conductivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thermal conductivity of most materials varies with temperature.
``heat_conduction`` expresses this
by multiplying a base conductivity, itself allowed to vary with position and
time, by a polynomial in the temperature:

.. math::
   :label: kpoly

   k(\mathbf{x}, t, T) = k_0(\mathbf{x}, t)
   \left( c_0 + c_1 T + c_2 T^2 + \cdots \right) .

The base value :math:`k_0` is the ``thermal_conductivity`` parameter: a
number, a function, an expression or the name of a property.  The
polynomial multiplies it in every case.
The coefficients are the ``temperature_polynomial`` list, evaluated by Horner's
rule from the highest downwards.  Two details of :eq:`kpoly` require attention.
The polynomial **multiplies** the base value, so a list supplied by the user
must carry its own constant term: the default list :math:`\{1\}` leaves the
conductivity unchanged, and a material with :math:`k = 20\,(1 + 0.01\,T)` is
written as a base value of :math:`20` together with the list
:math:`\{1, 0.01\}`.  Further, the argument of the polynomial is the
temperature variable itself, in the units of that variable, with no reference
temperature subtracted, so its coefficients must be converted along with the
rest of the problem if the scale changes.

For the solver, the consequence is that :eq:`energy` becomes nonlinear in
:math:`T`, because the flux contains the product of a function of :math:`T`
with :math:`\nabla T`.  A single linear solve is then insufficient, and the
choice of iteration determines where the temperature inside :eq:`kpoly` is
taken from.  Under Newton's method the polynomial is evaluated at the current
iterate as a differentiated quantity, so the Jacobian gains the term
:math:`(\partial k / \partial T) \nabla T` alongside the usual
:math:`k \nabla \psi`, and convergence is quadratic.  Under direct iteration the
polynomial is evaluated at the previous iterate, which is treated as a
constant, the matrix then resembles
that of a linear problem with a spatially varying conductivity, and convergence
is linear.  One accessor expresses the distinction: the conductivity is built
from ``coefficientValue``, which returns the current iterate in Newton mode and
the lagged one in Picard mode, while the gradient multiplying it is always the
current differentiated one.

Verification of the heat transfer module
----------------------------------------

The module is verified in ``tests/python/test_heat_transfer.py`` against
published dual mesh results [Reddy2024]_.  The cooling fin, a one-dimensional
problem with a Robin condition at the tip, is solved on ten and twenty elements,
and its nodal temperatures agree within :math:`6\times10^{-3}\ \mathrm{K}` and
the recovered heat flow at the base of :math:`4807\ \mathrm{W}` within three
parts in ten thousand.  The bus bar, which combines volumetric generation, two
fixed edges and a convecting top surface, reproduces every printed digit on a
:math:`10 \times 5` mesh.  The nonlinear plate with
:math:`k = k_0 (1 + 100\,T)` matches within three hundredths of a degree by
Newton's method and by direct iteration alike.  The size of this tolerance is
governed by the relative convergence criterion of :math:`10^{-3}` of the
published solution, and the discretization error is smaller.

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
of a Newtonian fluid, together with the body force.  A steady solve drops
:math:`\partial \mathbf{v} / \partial t`, and a transient solve keeps it (see
`Transient flow`_ below).  The
ratio of the convective to the viscous term defines the Reynolds number
:math:`Re = \rho V L / \mu` for a problem of characteristic speed :math:`V` and
length :math:`L`.

The pressure as a Lagrange multiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conduction gives one equation for one field, and that field appears in it.  The
Navier-Stokes system lacks this structure.  In two dimensions there are three
fields, :math:`u`, :math:`v` and :math:`P`, and three equations, two momentum
balances and continuity, but the pressure appears in the momentum equations and
nowhere in :eq:`continuity`.  No equation has the pressure as its principal
unknown, and no constitutive law delivers it from the kinematics.  In this
setting the pressure is the Lagrange multiplier enforcing the constraint
:eq:`continuity`, with no equation of state relating it to the density.  It
takes whatever value keeps the flow divergence free, and in a fully enclosed
flow it is fixed only to within an additive constant.

This has a strict consequence for a method that interpolates velocity and
pressure together.  The two interpolation spaces cannot be chosen
independently.  They must jointly satisfy the Ladyzhenskaya-Babuška-Brezzi, or
inf-sup, condition, which requires the pressure space to be sufficiently small
relative to the velocity space that every nonzero discrete pressure couples,
through the divergence, to some discrete velocity.  A pair violating it yields a
singular or nearly singular system and pressure fields contaminated by spurious
modes, the checkerboard oscillation being the best-known example.  Equal-order
interpolation of velocity and pressure, the natural choice for a library built
on a single family of Lagrange elements, violates this condition.

The penalty formulation
^^^^^^^^^^^^^^^^^^^^^^^

The library avoids the mixed problem by the penalty formulation
[Reddy2024]_.  In this formulation the constraint is relaxed and the fluid is
treated as very slightly compressible, through the constitutive assumption

.. math::
   :label: penalty

   P = - \gamma \, \nabla \cdot \mathbf{v} ,

in which :math:`\gamma` is the penalty parameter, a number with the units of
viscosity.  The sign follows from physical reasoning: a fluid element being
compressed has :math:`\nabla \cdot \mathbf{v} < 0` and must be at raised
pressure.  As :math:`\gamma \to \infty` a bounded pressure forces
:math:`\nabla \cdot \mathbf{v} \to 0` and the incompressible problem is
recovered.  At finite :math:`\gamma` the flow carries a residual dilatation of
order :math:`P / \gamma`.  The benefit is that :eq:`penalty` removes the pressure as
an unknown.  Substituting it into :eq:`momentum` expresses :math:`P` through the
velocity gradients and leaves two equations in the two velocity components, both
of the same second-order form as the conduction equation and hence expressible
in the canonical form :eq:`canonical_hf`.  Continuity is then imposed only
through the penalty term, which enforces it approximately, and no other part of
the framework requires knowledge of the constraint.

The choice of :math:`\gamma` involves a trade-off, and :math:`\gamma` is the
parameter most often set unsuitably.  It must be measured against the viscosity,
since the penalty term competes with the viscous term within the same equation,
and a ratio of roughly :math:`10^4` to :math:`10^7` is suitable, that is,
:math:`\gamma \approx 10^4 \mu` to :math:`10^7 \mu`.  The default
:math:`\gamma = 10^8` suits a viscosity of order unity.  At the low extreme the
constraint is weakly enforced, the computed flow is measurably compressible and
the solution corresponds to a different physical problem.  At the high extreme
the penalty contribution dominates every matrix entry, the viscous contribution
is lost to round-off error, and the system becomes so ill-conditioned that even a direct
factorization loses accuracy, so a further increase of :math:`\gamma` degrades
the velocity field.  A useful check is the residual divergence, which should
behave as :math:`|\nabla \cdot \mathbf{v}| \sim P / \gamma` and which the tests
confirm stays below :math:`10^{-5}` for the cavity at the default setting.

Reduced integration and locking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The penalty term must be integrated with a **reduced** quadrature rule, and the
formulation fails without this reduction.  The argument rests on counting
constraints and unknowns.  Integrating the penalty term with a rule of
:math:`n` points per element imposes, in the limit of large :math:`\gamma`, the
discrete constraint :math:`\nabla \cdot \mathbf{v} = 0` at each of those points,
so a mesh of :math:`N_e` elements carries :math:`n N_e` constraints.  Against
this stands the number of velocity degrees of freedom: a structured mesh of
four-node quadrilaterals has about as many nodes as elements, hence about
:math:`2 N_e` velocity unknowns in two dimensions, fewer once the wall
conditions are applied.  With the full :math:`2 \times 2` Gauss rule,
:math:`n = 4` and there are some :math:`4 N_e` constraints against
:math:`2 N_e` unknowns.  A system with twice as many constraints as unknowns has
in general only the trivial solution.  The computed response is accordingly far
too stiff, and in the limit the velocity field is identically zero.  This failure
is locking, and refining the mesh does not remove it, because refinement
multiplies constraints and unknowns in the same proportion.

The one-point rule sets :math:`n = 1`, giving :math:`N_e` constraints against
about :math:`2 N_e` unknowns, a ratio of two unknowns per constraint.  That is
the ratio the continuum problem itself has, since at every point of a
two-dimensional flow two velocity components are subject to one incompressibility
condition, and it is the value at which the discrete problem neither locks nor
leaves the constraint unenforced.  ``penalty_incompressibility`` accordingly
defaults to the ``midpoint`` rule and to ``reduced_integration = true``, both of
which differ from the framework's general defaults, the latter meaning that the
velocity gradients are evaluated at the element centroid while the geometric integration
proceeds as usual.  This is selective reduced integration: the viscous term
keeps its full rule and only the constrained term is under-integrated.  A fuller
rule on the penalty term is available, and its only appropriate use is to
demonstrate the locking.

The same counting decides how a discretisation must evaluate the term.  A
gradient taken as a two-point difference along each edge would impose the
constraint once per edge, which exceeds the one constraint per element of the
reduced rule and locks the flow: on a lid-driven cavity the velocities were
five orders of magnitude too small.  Both methods of dualmesh evaluate the
penalty term with the interpolated gradient at the element centroid.

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
  :math:`\mathbf{F} = \mu ( \nabla u_i + (\nabla \mathbf{v})_i )` and no source.
  Its :math:`d`-th component is
  :math:`\mu ( \partial u_i / \partial x_d + \partial u_d / \partial x_i )`, so
  in two dimensions the :math:`u` equation carries
  :math:`(2\mu\,\partial u/\partial x,\ \mu(\partial u/\partial y +
  \partial v/\partial x))`.  The viscosity is evaluated from position and time
  only, which makes this kernel Newtonian.  A shear-rate dependent viscosity
  requires a property object and a kernel written for the purpose.
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
  current differentiated one.  A Stokes flow is obtained by omitting this
  kernel, and the density then plays no role.
* ``body_force``, if present, contributes no flux and the source
  :math:`S = -f_i`, with the sign chosen so that a positive intensity increases
  the velocity.

Each object takes the same ``velocities`` list, in coordinate order, and a
zero-based ``component`` index into it.  The index is range checked, but its
agreement with the ``variable`` parameter is not, so a mismatch assembles one
momentum equation into another without any error message.  The
``fluid_dynamics`` physics prevents such a mismatch by adding the viscous
and penalty kernels for every component, the
inertia kernel when a non-zero density is given, and the pressure property
object described next.

The natural boundary quantity of :eq:`penalty_momentum` follows the same rule as
in conduction:

.. math::

   \mathbf{n} \cdot \mathbf{F}
   = \sum_d n_d \, \mu \left( \frac{\partial u_i}{\partial x_d}
     + \frac{\partial u_d}{\partial x_i} \right) - P \, n_i ,

where :eq:`penalty` replaces :math:`\gamma \nabla \cdot \mathbf{v}` by
:math:`-P`.  This is the :math:`i`-th component of the surface traction, so a
boundary with no condition at all is traction free, which is the usual outflow
condition.  Walls and moving lids are imposed as
``Dirichlet`` conditions on the velocity components.

Recovering the pressure
^^^^^^^^^^^^^^^^^^^^^^^

The pressure is eliminated from the unknowns but remains available, because
:eq:`penalty` recovers it from the converged velocity field.  The
``penalty_pressure`` property object declares the property ``pressure`` and evaluates
:math:`P = -\gamma \nabla \cdot \mathbf{v}` wherever it is requested.  Two
cautions apply.  First, the value of :math:`\gamma` given to the property object must
be identical to the one given to ``penalty_incompressibility``.  The two objects
hold separate copies, nothing checks that they agree, and a mismatch scales the
recovered pressure by the ratio of the two, giving a plausible looking field of
the wrong magnitude. Second, the pressure should be evaluated at the element
centroids, which are the reduced-order points at which the constraint is
imposed.  The divergence there is the quantity the formulation controls, and the
recovered pressure is markedly more accurate there than at the nodes.

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

As the Reynolds number rises, the domain of quadratic convergence shrinks.  At
:math:`Re = 1000` in the lid-driven cavity the recirculating flow is far from
the rest state, and an iteration started from zero velocity takes a first step
so large that it leaves the basin of attraction and diverges.  The remedy is
**load stepping**: the boundary velocity is applied in a sequence of load
factors, each step solved to convergence and its solution used as the initial
guess for the next.  The cavity test uses the
factors :math:`0.1, 0.25, 0.5, 0.75, 1.0` and marks the lid condition with
``scale_with_load`` so that it is the quantity ramped.  Each intermediate
problem is then close enough to its predecessor for Newton's method to converge,
and the computation passes through a sequence of flows of increasing Reynolds
number up to the final one.

**The Stokes start.**  A steady flow whose velocity is at rest (zero at every
unknown that no condition prescribes) starts Newton's method from the Stokes
solution: the same problem with the inertia :math:`\rho (\mathbf{v} \cdot
\nabla) \mathbf{v}` turned off, solved at the first load factor.  The state at
rest holds the prescribed boundary velocities next to zero values, so its
linearised inertia has steep gradients, and the first step from it can leave the
basin of convergence.  On a channel at :math:`Re = 100` with the inflow
:math:`y` or :math:`6 y (1 - y)`, a cell-centred finite volume method, which
dualmesh had until 24 October 2026, landed its first step from rest 300 away
from the Stokes pressure, whose range is 0 to 28, and Newton's method diverged.
From the Stokes solution Newton's method converges in a few iterations to the
solution that continuation in the Reynolds number reaches.  The Stokes solve
takes the pressure mass preconditioner when the flow takes ``"LSC"`` or
``"PCD"``, which approximate a convection that it does not have.  A velocity
that the user gives as the start (``set_values`` or ``initial_condition``) is
kept, and the report of the solve states the Stokes start.

**A diverging iteration.**  When the residual grows to a million times its
first value, the solve stops and says that Newton's method diverged, with the
remedies: load steps, a closer start, Picard iteration, or an advection scheme
for a flow at a high cell Reynolds number.  Before, the iteration went on until
its matrix could not be factorised, and the error then pointed at the boundary
conditions.

Direct iteration with relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The alternative is direct, or Picard, iteration.  Every nonlinear coefficient is evaluated at the previous iterate,
so the transporting velocity in :math:`\rho (\mathbf{v} \cdot \nabla) u_i` is
held fixed and each iteration is a linear convection-diffusion solve.  The
matrix is cheaper to form and the method is far less sensitive to the starting
point, but convergence is linear, and at higher Reynolds number the iteration
tends to overshoot and oscillate between two states without converging.  The
remedy is under-relaxation: after the linear solve produces a tentative iterate
:math:`\tilde{\mathbf{U}}^{r+1}`, the code combines it with the previous one,

.. math::
   :label: relaxation

   \mathbf{U}^{r+1} = \left( 1 - \beta \right) \tilde{\mathbf{U}}^{r+1}
   + \beta \, \mathbf{U}^{r} ,

where :math:`\beta` is the ``relaxation`` parameter, not to be confused with
the penalty parameter :math:`\gamma` of :eq:`penalty`.  The weight :math:`\beta` multiplies the **old** iterate:
:math:`\beta = 0`, the default, is the unrelaxed method, a value near one leaves
the iterate almost unchanged, and a value of about one half is suitable for a
flow that converges with difficulty.  Relaxation is skipped on the first
iteration of each load step, so the method takes one full step before it begins
damping.  The same mechanism serves both nonlinear solvers, although it is
ordinarily required only by direct iteration.

The two strategies address the same difficulty in different ways.  Newton's
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
but it has three costs.  First, the penalty parameter :math:`\gamma` must be
large for the divergence to be small, which makes the matrix badly conditioned.
Second, the penalty term must be integrated with a reduced rule.  Third, the pressure is
only recovered afterwards, element by element.  The *pressure-velocity* (or *mixed*)
formulation, selected with ``formulation="pressure"`` of the
``fluid_dynamics`` physics, keeps the pressure :math:`p` as
an unknown field and solves the mass equation alongside the momentum
equations.  It works with both methods.

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
the viscous part of :eq:`pv_momentum`, the kernel ``pressure_gradient`` the
pressure part :math:`-p\,\mathbf{e}_i`, and ``mass_conservation`` the mass
equation, which is attached to the pressure variable.  Because the pressure
is part of the flux, the natural boundary quantity of a momentum equation is
the full traction :math:`(\boldsymbol{\sigma} \mathbf{n})_i`, and a boundary
without a velocity condition is a traction-free outlet.  In axisymmetric
coordinates without swirl the radial equation also carries the hoop stress:
the source gains :math:`-\sigma_{\theta\theta}/r = -(2 \mu u_r / r - p)/r`,
which ``viscous_stress`` and ``pressure_gradient`` add between them.

Why equal-order interpolation needs stabilisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The pressure in an incompressible flow has no governing equation of its own.
It is the Lagrange multiplier of the constraint
:math:`\nabla \cdot \mathbf{v} = 0`, and a discretisation determines it only if
the discrete velocity space is rich enough that every discrete pressure mode
couples to some discrete velocity.
This is the inf-sup, or Ladyzhenskaya-Babuška-Brezzi, condition.  Interpolating
velocity and pressure with the same functions violates it: on a quadrilateral
mesh the checkerboard pressure, alternating :math:`+1` and :math:`-1` from node
to node, produces no force on any discrete velocity and therefore leaves the
discrete equations unchanged.  Two remedies exist.  One is to enrich the
velocity, as the Taylor-Hood element does with quadratic velocity and linear
pressure.  The other, used here, is to *stabilise*: to add to the mass equation
a term that controls the pressure gradient and vanishes for the exact solution.

The stabilised mass flux
^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mathbf{r}_m` be the residual of the momentum equation in its
strong form,

.. math::
   :label: pv_residual

   \mathbf{r}_m = \rho \left( \frac{\partial \mathbf{v}}{\partial t}
   + (\mathbf{v} \cdot \nabla) \mathbf{v} \right) + \nabla p - \mu \Delta \mathbf{v}
   - \mathbf{f} .

``mass_conservation`` replaces the mass flux :math:`-\mathbf{v}` by

.. math::
   :label: pv_mass_flux

   \mathbf{F} = \tau\, \mathbf{r}_m - \mathbf{v} .

Quadratic elements give the viscous term from the second derivatives of their
shape functions, :math:`\mathbf{H}_x N = \mathbf{A} (\mathbf{H}_\xi N -
\sum_c \partial N/\partial x_c\, \mathbf{H}_\xi x_c) \mathbf{A}^{\mathsf{T}}`
with :math:`\mathbf{A} = \mathbf{J}^{-\mathsf{T}}`, which holds for curved
elements too, plus the terms of the axisymmetric and spherical Laplacians.  The
residual of the exact solution is then zero, and the stabilisation is
consistent: Poiseuille flow, whose velocity does not depend on :math:`x` and
whose pressure is linear, is exact to round-off at the nodes on ``Quad9`` and
``Tri6`` elements with both methods (``tests/python/test_poiseuille.py``).

Inside a linear element the Laplacian of the interpolant vanishes, so the
residual of linear elements leaves the viscous term out, as MOOSE and COMSOL
do.  The residual of the exact solution is then :math:`-\mu \Delta \mathbf{v}`,
and the stabilisation adds an error of order :math:`h^2` in the velocity.  On
Poiseuille flow with 8, 16 and 32 cells across, the largest nodal error of the
velocity is :math:`1.8 \times 10^{-2}`, :math:`4.4 \times 10^{-3}` and
:math:`1.1 \times 10^{-3}`, and that of the pressure 1.2, 0.48 and 0.22, first
order, with the largest values next to the walls.  Two remedies were measured
on 25 October 2026.  The recovered Laplacian of [JansenCollis1999]_ and the orthogonal
subscales of [Codina2002]_ lowered the error of the Kovasznay flow by a third,
and the orthogonal subscales made Poiseuille flow exact.  The orthogonal
subscales were not better in every case: the cavity at :math:`Re = 1000` was
closer without them.  Both cost more than the refinement that gives the same
gain.  The recovered Laplacian couples each node to two rings of neighbours,
and lagged it took 3 to 10 times the Newton iterations.  The orthogonal
subscales need a projection of the residual, lagged (8 to 30 times the
iterations) or as unknowns.  For the Galerkin
finite element method, integrating :math:`\nabla \psi \cdot \mathbf{F}` over
the domain gives the mass equation plus
:math:`\int \tau \nabla \psi \cdot \mathbf{r}_m\,\mathrm{d}\Omega`, which is exactly
the pressure-stabilising Petrov-Galerkin (PSPG) method of
[HughesFrancaBalestra1986]_.  Its pressure-gradient part,
:math:`\int \tau \nabla \psi \cdot \nabla p\,\mathrm{d}\Omega`, is a small pressure
Laplacian that removes the checkerboard mode.  For the control volume methods
the same flux is integrated over the faces of each control volume, and the
face mass flux becomes the interpolated velocity corrected by
:math:`\tau (\nabla p - \dots)`.  This is the idea behind the momentum
interpolation of [RhieChow1983]_, which finite volume codes use to avoid
checkerboard pressures on colocated grids, here extended from the pressure
gradient to the whole momentum residual.

The parameter :math:`\tau` has the units of a time divided by a density, so
that :math:`\tau \mathbf{r}_m` is a velocity.  It is the parameter of
[Tezduyar1991]_ (see also [TezduyarMittalRayShih1992]_), written per unit
density so that it also covers the Stokes equations:

.. math::
   :label: pv_tau

   \tau = \left[ \left( \frac{2 \rho}{\Delta t} \right)^2
   + \left( \frac{2 \rho |\mathbf{v}|}{h} \right)^2
   + 9 \left( \frac{4 \mu}{h^2} \right)^2 \right]^{-1/2} .

The three terms are the reciprocal time scales of the unsteadiness, of
convection across an element and of diffusion across it.  The factor 9 on the
viscous term, which the navier_stokes module of MOOSE [MOOSE2025]_ takes
(``INSADTauMaterial``), gives the viscous limit :math:`h^2 / (12 \nu)`, that of
the optimal parameter of linear elements, :math:`(h / 2 |\mathbf{v}|)(\coth
\mathrm{Pe}_h - 1 / \mathrm{Pe}_h)` [BrooksHughes1982]_, where the form of
Tezduyar without it gives :math:`h^2 / (4 \nu)`, three times too much.
Measured on 32 by 32 elements at :math:`Re = 100`, against a Taylor-Hood
solution on :math:`64 \times 64` elements, the factor halves the error of the
cavity centreline velocities (``FEM`` 0.0045 and 0.0088 against 0.0016 and
0.0041), and it lowers the error of the Kovasznay flow by about a quarter for
every method and that of the Nusselt number of natural convection at
:math:`Ra = 10^5` from 1.1 % to 0.5 %.  The first is absent
in a steady solve.  The element size is :math:`h = (c V_e)^{1/d}`, with
:math:`V_e` the element volume and :math:`c = 1` for quadrilaterals and
hexahedra, 2 for triangles and prisms and 6 for tetrahedra and pyramids, so
that every element of a structured grid of spacing :math:`h` is assigned the
size :math:`h`.  The size is halved for a quadratic element.  With inertia,
``momentum_stabilization`` adds the streamline-upwind term of
[BrooksHughes1982]_ to each momentum equation as the flux
:math:`\rho \tau\, r_{m,i}\, \mathbf{v}`, which damps the oscillations of
convection-dominated flow.  The ``fluid_dynamics`` physics adds it
whenever the density is non-zero.

Every force in the momentum equations must also appear in
:math:`\mathbf{r}_m`, because a force missing there causes the stabilisation
term to act against that force.  The ``fluid_dynamics`` physics passes
its ``body_force`` and the buoyancy of a ``nonisothermal_flow`` coupling to the
stabilisation, with the
same load scaling as the force itself.  Without the buoyancy in
:math:`\mathbf{r}_m` the natural convection benchmark fails at high Rayleigh
number, because there :math:`\tau \nabla p` is much larger than the velocity.

The omitted viscous term, :math:`-\nabla \cdot (\mu \nabla \mathbf{v})`, needs
second derivatives, which the kernels do not have.  It vanishes on linear
simplices and on bilinear elements that are parallelograms, and omitting it
keeps the optimal first order of the energy norm.  On quadratic elements it
does not vanish, the omission becomes an inconsistency, and the velocity
error converges only at second order in :math:`L^2`, one order below the
optimal third order.  For quadratic elements the Taylor-Hood element, which
needs no stabilisation, is the appropriate choice.

The Taylor-Hood element
^^^^^^^^^^^^^^^^^^^^^^^

The other remedy for the checkerboard is to make the velocity space richer than
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
would be inconsistent and the velocity would lose its third order.

**First-order variables on a quadratic mesh.**  A variable declared
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
norms and every other post-processing routine obtain the correct field with no
treatment specific to the two orders.  Inside an element the local degrees of
freedom are numbered compactly (every variable at the corners, then only the
quadratic ones at the other nodes), and the automatic differentiation seeds
one derivative per local degree of freedom: 22 on a ``Quad9`` element, against
27 for three variables at every node, and :math:`27 \times 3 + 8 = 89` on a
``Hex27`` element.

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
directly where that is inexpensive, and splits a larger one with the least
squares commutator, a block preconditioner that approximates the Schur
complement of the pressure (see :doc:`/user_guide/solving`).  In three
dimensions the direct factorisation grows costly quickly (a ``Tet10`` mesh
with about 20 000 unknowns takes about :math:`4\ \mathrm{s}` per
factorisation in one thread), and on a three-dimensional cavity on two
processes the split is about twice as fast as MUMPS from 37 000 unknowns on.

**Verification.**  ``tests/python/test_taylor_hood.py`` checks the element in
four ways.

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
entity that fixes the pressure level (see below).  The Taylor-Hood element, which has no pressure Laplacian to
carry the imbalance, shows no spike, and its pressure still converges at second
order.

*The inf-sup condition itself*, by the numerical test of
[ChapelleBathe1993]_ (Proposition 2.2, Eq. (14)), on meshes of increasing
refinement as they recommend (Sect. 2.5).  With :math:`\mathbf{A}` the vector Laplacian (whose
energy is :math:`|\mathbf{v}_h|_1^2`), :math:`\mathbf{B}` the discrete
divergence and :math:`\mathbf{M}` the pressure mass matrix, for velocities
that vanish on the boundary, :math:`\beta_h^2` is the smallest non-zero
eigenvalue :math:`\lambda` of

.. math::

   \mathbf{B} \mathbf{A}^{-1} \mathbf{B}^T \mathbf{q} = \lambda\, \mathbf{M} \mathbf{q}

(the smallest eigenvalue is zero, for the constant pressure that an enclosed
flow does not determine).  On the unit square:

.. list-table:: Discrete inf-sup constant :math:`\beta_h` on an :math:`n \times n` mesh
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
diagonal, which leaves two corner triangles with every vertex on the boundary,
and the constant remains bounded nevertheless.  The same computation checks that the
pressure coupling of the momentum equations is the negative transpose of the
divergence to round-off, as the Galerkin method requires.

*Newton's method* converges quadratically (the exact Jacobian includes the
mixed-order assembly), and the *lid-driven cavity* at :math:`Re = 100` on
:math:`16 \times 16` ``Quad9`` elements agrees with [Ghia1982]_ on both
centrelines within 0.01.

*Misuse is refused* with a message that states the required correction, in the
following cases:

* a first-order variable with any method other than ``FEM``,
* ``formulation="taylor_hood"`` on a linear mesh or with another method,
* stabilisation or SUPG requested with this element,
* an existing pressure of the wrong order, and
* ``mass_conservation`` without stabilisation on an equal-order pressure, which
  would otherwise return the spurious modes.

Boundary conditions for the pressure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mass equation requires the flow through the boundary.
``mass_flux`` supplies it as the natural boundary quantity
:math:`q = -\mathbf{v} \cdot \mathbf{n}`, computed from the discrete velocity,
and the ``fluid_dynamics`` physics places it on every side set.  For the finite
element method this reproduces the boundary term of PSPG exactly.  Summing the
mass equations of all control volumes gives
:math:`\oint \mathbf{v}_h \cdot \mathbf{n}\,\mathrm{d}s = 0`, so the discrete flow
conserves mass globally, whatever the stabilisation.

An enclosed flow, with a velocity condition on every boundary, determines the
pressure only up to a constant.  The solver finds such a flow from its Jacobian
:math:`\mathbf{J}` on the first Newton iteration.  A constant pressure
:math:`\mathbf{1}_p` is then a solution of the homogeneous problem, with
:math:`\|\mathbf{J}\,\mathbf{1}_p\| \le 10^{-9}\, \||\mathbf{J}|\,\mathbf{1}_p\|`.
This test covers every boundary condition, including the symmetry planes, which
hold the normal velocity and leave the pressure free.  The solver then holds the
unknown of the pressure nearest the centre of the fluid at its value (an
unknown in a corner slows the Krylov solvers by about 40 %).  After each
converged solve it shifts the pressure to a mean of zero over the fluid and
takes the residual, and so the reactions, at the shifted state.

The shifted residual also checks that the level is still free at the solution.
A viscosity, a density or a source that depends on the pressure makes the level
change the flow, which a start at rest may not show.  The mass equations of an
enclosed flow then still sum to the boundary flux whatever the solution
(:math:`\mathbf{J}^{\mathsf{T}} \mathbf{1}_p = 0`), so they do not fix the
level, and each level gives a flow of its own.  The solve stops with an error
that asks for the pressure at a point (``point_Dirichlet``) or a pressure or
traction condition.

The held unknown replaces one mass equation.  The held entity is therefore the
only place where a net boundary flux
:math:`\oint \mathbf{v}_h \cdot \mathbf{n}\,\mathrm{d}s \neq 0` can be accommodated.
With walls and a sliding lid the discrete net flux is zero, the held unknown
introduces no error, and the result does not depend on which unknown is held or
on the number of processes.  With prescribed velocities that cross the boundary,
however, the nodal interpolant of the data carries a net flux of order
:math:`h^2`, and the pressure Laplacian of the stabilisation, whose coefficient
is :math:`\tau \sim h^2`, converts it into a pressure spike of order one at the
held entity.  The solve then warns with the size of the net flux.  Such a flow
should have an outlet: a boundary with a traction condition (or none)
determines the pressure.

Verification of the pressure-velocity formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tests/python/test_pressure_velocity.py`` checks the formulation with the
manufactured solutions of the ``fluid_dynamics`` physics
(:class:`dualmesh.mms.ManufacturedSolution`), whose exact
velocity is divergence free, whose right boundary is an outlet with the exact
traction and whose pressure has no symmetry.  With the density
:math:`\rho = 1` and the viscosity :math:`\mu = 1`, the observed orders on the
finest pair of meshes (:math:`32^2` and :math:`64^2` elements, or :math:`8^3`
and :math:`16^3` in three dimensions) are:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Mesh
     - :math:`\|\mathbf{v} - \mathbf{v}_h\|_{L^2}`
     - :math:`|\mathbf{v} - \mathbf{v}_h|_{H^1}`
     - :math:`\|p - p_h\|_{L^2}`
   * - Distorted Quad4
     - 1.98 to 1.99
     - 1.01
     - 1.60 to 1.65
   * - Distorted Tri3
     - 1.99
     - 1.00 to 1.01
     - 1.59 to 1.60
   * - Axisymmetric Quad4 with the axis, both methods
     - 1.95 to 1.97
     - 1.03
     - 1.61 to 1.68
   * - Hex8, both methods
     - 1.81 to 1.83
     - 1.15 to 1.21
     - 1.59 to 1.66

The theory of the stabilised methods guarantees first order for the velocity
in :math:`H^1` and for the pressure in :math:`L^2`
[HughesFrancaBalestra1986]_, and second order for the velocity in :math:`L^2`.
All three are observed or exceeded.  The pressure is not controlled in
:math:`H^1`, and its :math:`H^1` error converges only at about half an order,
which is the behaviour usually reported for equal-order linear elements
[BrezziPitkaranta1984]_.

The physics is checked against two benchmarks.  For the lid-driven cavity at
:math:`Re = 100` on a :math:`32 \times 32` mesh, both methods reproduce the
centreline velocities of [Ghia1982]_ to within 0.015, and to within 0.0074 on
:math:`64 \times 64`.  At :math:`Re = 1000` on :math:`64 \times 64` the finite
element method gives the minimum horizontal velocity :math:`-0.381`, against
:math:`-0.383` in the reference.  Both methods require the two corner
nodes of the lid to belong to the walls.  With the lid velocity prescribed at
those nodes, flow crosses the side walls through the first row of elements, and
the vortex is noticeably too weak at :math:`Re = 1000`.  For natural convection,
with the buoyancy in the stabilisation, the Nusselt number and velocity maxima
on the :math:`32 \times 32` graded mesh are within 2 % of [DeVahlDavis1983]_ up
to :math:`Ra = 10^6`.  Newton's method converges quadratically with both
methods.

Transient flow
--------------

A transient solve keeps the inertia :math:`\rho\, \partial \mathbf{v} / \partial t`
of :eq:`momentum`, which ``fluid_dynamics`` adds to every momentum equation
when the density is positive, and advances the flow in one of two ways, chosen by
``time_integration``.

``"monolithic"``, the default, applies the theta method of ``solve_transient`` to
the whole system: every step solves the velocity and the pressure together by
Newton's method, as a steady solve does.  It is the robust choice for strongly
coupled problems and for large steps, at the price of a non-symmetric saddle point
system at every step.

The mass equation has no time derivative, so every :math:`\theta` takes it at
the new time with the weight 1, and so it takes the pressure gradient and the
stabilisation of the momentum equations (``momentum_stabilization``) and of
the mass flux.  Their residual :eq:`pv_residual` is then the residual of the
discrete momentum equation, the time difference and :math:`\nabla p^{n+1}`
with the other terms weighted by :math:`\theta` at the new state and by
:math:`1 - \theta` at the old one, so it vanishes for the solution of the
discrete equations.  On :math:`\mathbf{v} = (1 + t, 0)`, :math:`p = -x`,
which the elements hold exactly, Crank-Nicolson is now exact to round-off.
Weighted between the two times, the pressure was an average of two steps, its
error came back with the opposite sign at every step and grew, to
:math:`5 \times 10^{-2}` after four steps on :math:`16 \times 16` elements.
On the rotating flow :math:`\mathbf{v} = (1 + t)(y, x)` Crank-Nicolson has the
error of backward Euler.

``"projection"`` is the splitting of Nek5000 and nekRS [Fischer2021]_, which
replaces the saddle point system by symmetric positive definite problems, each
with its own fast solver.  With the kinematic pressure :math:`P = p / \rho` and
the kinematic viscosity :math:`\nu = \mu / \rho`, the step to the time
:math:`t^n` collects the known terms in

.. math::

   \mathbf{u}^* = -\sum_{j=1}^{k} \left[ \beta_j \mathbf{v}^{n-j}
     + \Delta t\, \alpha_j (\mathbf{v}^{n-j} \cdot \nabla) \mathbf{v}^{n-j} \right]
     + \frac{\Delta t}{\rho} \mathbf{f}^n ,

where :math:`\beta_0, \dots, \beta_k` are the coefficients of the backward
difference of order :math:`k` (BDF\ :math:`k`) and :math:`\alpha_1, \dots,
\alpha_k` those of the extrapolation of the same order (EXT\ :math:`k`) of the
advection, with :math:`k` the ``time_order``.  The coefficients are computed from
the time levels themselves, as the derivative and the value at :math:`t^n` of
Lagrange polynomials, so that a step shortened to land on an output time keeps
the order.  The momentum equation at :math:`t^n` then reads

.. math::

   \frac{\beta_0}{\Delta t} \mathbf{v}^n = \frac{\mathbf{u}^*}{\Delta t}
     - \nabla P^n + \nu \nabla^2 \mathbf{v}^n .

**The pressure.**  Testing this equation with the gradient of a pressure test
function :math:`q`, and writing the viscous term of a divergence-free velocity as
:math:`\nu \nabla^2 \mathbf{v} = -\nu \nabla \times \boldsymbol{\omega}`, with the
vorticity :math:`\boldsymbol{\omega} = \nabla \times \mathbf{v}` extrapolated from
the steps before, gives the pressure equation

.. math::

   (\nabla q, \nabla P^n) = \frac{1}{\Delta t} (\nabla q, \mathbf{u}^*)
     - \frac{\beta_0}{\Delta t} \int_{\Gamma} q\, \mathbf{n} \cdot \mathbf{v}_b \,\mathrm{d}\Gamma
     - \nu \int_{\Gamma} \mathbf{n} \cdot (\boldsymbol{\omega} \times \nabla q) \,\mathrm{d}\Gamma ,

where :math:`\mathbf{v}_b` is the velocity on the boundary, the prescribed one
where it is prescribed.  The first boundary term is
:math:`(\beta_0 / \Delta t)(\nabla q, \mathbf{v}^n)` of the divergence-free
velocity.  The second is the curl-curl form of the viscous term, which
[Fischer2021]_ keeps to control the divergence error at the boundary, written as
an integral over the boundary by
:math:`(\nabla q, \nabla \times \boldsymbol{\omega}) = \int_\Gamma \mathbf{n} \cdot
(\boldsymbol{\omega} \times \nabla q) \,\mathrm{d}\Gamma`, which holds because
:math:`\nabla \cdot (\boldsymbol{\omega} \times \nabla q) = \nabla q \cdot \nabla
\times \boldsymbol{\omega}`.  In this form it needs only the first derivatives of
the velocity, so it applies to linear elements, whose second derivatives vanish.
Without it the velocity of the vortex below converges at about first order, and
with it at second order.  An enclosed flow, with no prescribed pressure, has its pressure fixed at
one node, after the right-hand side is made to sum to zero.

**The velocity.**  Each component then solves the Helmholtz equation

.. math::

   \nu (\nabla w, \nabla v_i^n) + \frac{\beta_0}{\Delta t} (w, v_i^n)
     = \frac{1}{\Delta t} (w, u^*_i) - \left(w, \frac{\partial P^n}{\partial x_i}\right) ,

with the prescribed velocity on its boundary and the natural condition
:math:`\nu\, \partial v_i / \partial n = 0` where the velocity is free and the
pressure is prescribed (an outflow).

**The two methods.**  The equations above are written with test
functions.  The finite element method takes the shape functions.  The dual mesh
control domain method takes the indicator of the control domain :math:`\Omega_I` of each node, so that
:math:`(w, f)` becomes :math:`\int_{\Omega_I} f \,\mathrm{d}V` and
:math:`(\nabla q, \mathbf{F})` becomes
:math:`-\oint_{\partial \Omega_I} \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S` over
the interfaces between control domains, as in the assembly of the method.  The
mass and gradient terms are then integrals over the control domains, the
Laplacian is the flux of the gradient out of them, and
:math:`(\nabla q, \mathbf{u}^*)` is the flux of :math:`\mathbf{u}^*`.  The curl-curl term
keeps the form above with the shape functions: on linear elements, where the
vorticity is constant in each element, it equals the integral of
:math:`\mathbf{n} \cdot \nabla \times \boldsymbol{\omega}` over the part of the
boundary of each control domain.  The methods of the dual mesh integrate with
the midpoint rule of every part of a control domain and of every interface,
which is second order like the methods themselves: two points per direction give
the same errors at 2.4 times the cost.

**Equal-order elements at small steps.**  The term
:math:`(\nabla q, \mathbf{u}^*) / \Delta t` contains the divergence of the
velocities of the steps before, weighted by :math:`\beta_0 / \Delta t`, so the
splitting acts on the pressure as the stabilization of the pressure-velocity
formulation would, with the parameter :math:`\Delta t / \beta_0`.  That parameter
vanishes with the step, and equal-order elements, which are not inf-sup stable,
then lose their pressure stability: on the vortex below, the pressure error grows
from :math:`2.3 \times 10^{-2}` at :math:`\Delta t = 0.1` to
:math:`9.0 \times 10^{-2}` at :math:`\Delta t = 0.002` on the same mesh.  The weight
is therefore capped at :math:`1 / \tau`, with the parameter :math:`\tau` of
Tezduyar [Tezduyar1991]_ that the pressure-velocity formulation uses, without its
unsteady term.  Where :math:`\Delta t / \beta_0` exceeds :math:`\tau` the scheme
is that of [Fischer2021]_, and as :math:`\Delta t` goes to zero it tends to the
stabilized discretization of the monolithic solver.  The capped term multiplies a
divergence, so it does not change the exact solution, and the errors of the
vortex stay at :math:`1.3 \times 10^{-2}` (velocity) and
:math:`1.7 \times 10^{-2}` (pressure) for every step from 0.1 to 0.002.

**The solvers.**  The mass, stiffness and gradient matrices and the geometry of
every quadrature point are computed once, as Nek5000 keeps its geometric
factors, and the right-hand sides are assembled at every step on several
threads.  The pressure equation is solved with algebraic multigrid (hypre
BoomerAMG), starting from the combination of the earlier pressures (kept as up
to eight directions) that is closest to the solution, the projection-based
initial guess of [Fischer2021]_, introduced in [Fischer1998]_.  Each Helmholtz
equation is solved with a Jacobi preconditioner.  The matrices of the finite
element method are symmetric: their Krylov method is the conjugate gradient
method, and the closest combination is that in the energy norm.  Those of the
dual mesh are not symmetric in general: their Krylov method is GMRES, and the
closest combination is that of the smallest residual.  The matrices and the preconditioners are kept as long as
:math:`\beta_0 / \Delta t` is unchanged.  The ``"cfl"`` stepper of
``solve_transient`` chooses each step to keep the Courant number
:math:`\Delta t \max_e |\mathbf{v}| / h_e` at ``courant_number``, 0.5 by default,
the value [Fischer2021]_ uses with BDF3 and EXT3.

**Verification.**  ``tests/python/test_projection.py`` and
``dualmesh_parallel_tests`` check the following for both methods.

* The decaying Taylor-Green vortex,
  :math:`\mathbf{v} = (-\cos x \sin y, \sin x \cos y)\, e^{-2 \nu t}`, with its
  velocity prescribed on the walls of :math:`[0, 2\pi]^2`, :math:`\nu = 0.1`, BDF2
  and :math:`\Delta t \propto h`: the largest velocity error at :math:`t = 0.5`
  falls as :math:`6.2 \times 10^{-2}`, :math:`1.3 \times 10^{-2}`,
  :math:`3.0 \times 10^{-3}` and :math:`7.3 \times 10^{-4}` on
  :math:`8^2` to :math:`64^2` ``Quad4`` elements with the finite element method,
  :math:`7.6 \times 10^{-2}` to :math:`1.1 \times 10^{-3}` with the dual mesh
  control domain method, both at second order, and at second order on ``Tri3``
  elements as well.  In a periodic
  box the velocity converges at second order or faster (about third on these
  uniform grids) and the pressure at second order.
* The stagnation flow :math:`\mathbf{v} = g(t)\,(x, -y)`, whose linear velocity
  and quadratic pressure ``Quad9`` elements represent exactly, so that its error
  is that of the time integration alone: the pressure converges at orders 0.96,
  2.03 and 2.87 with BDF1, BDF2 and BDF3, with every method.  On the vortex with
  ``Quad9`` elements and the finite element method
  the velocity converges at first order with BDF1 and at second order with BDF2
  and BDF3, whose first steps take the orders below it.
* The distributed runs give the serial velocity, and the pressure up to its
  constant, to :math:`10^{-11}` on one to four processes, with walls and in a
  periodic box.

On the periodic vortex of :math:`256^2` ``Quad4`` elements at a Courant number
of one half (``dualmesh_benchmark projection``), each step takes one iteration of
the pressure solve, four without the projection of the earlier pressures, and two
of each velocity solve, with every method.  A step of the finite element method
takes 0.030, 0.016 and 0.010 s on one, two and four processes of a laptop, and a
step of the dual mesh control domain method 0.041 s on one,
because it evaluates the flow at the control domains and at their interfaces.

Verification of the fluids module
---------------------------------

The module is checked in ``tests/python/test_fluids.py`` against published
dual mesh penalty results [Reddy2024]_.  The creeping flow squeezed between two
approaching plates is computed on a graded :math:`20 \times 16` mesh and
reproduces the horizontal velocity profiles at two stations to within
:math:`2 \times 10^{-3}`.  The same test takes the recovered penalty pressure at
the element centroids and confirms that it follows Nadai's approximate solution
:math:`P = 3 \mu V_0 (a^2 + y^2 - x^2) / (2 b^3)` in the interior, with a
correlation above :math:`0.999` and a magnitude correct to within 5 %,
the comparison being restricted to the interior because the corners of this
problem are singular.  The lid-driven cavity is computed on a
:math:`16 \times 20` mesh graded towards the lid and reproduces the centreline
profile to within :math:`2 \times 10^{-3}` at :math:`Re = 0`,
where the flow is Stokes, and again at :math:`Re = 1000`, where the same profile
is obtained twice: once by Newton's method with the lid ramped in five load
steps, and once by relaxed direct iteration with :math:`\beta = 0.5`.  A further
test at :math:`Re = 400` confirms that the two nonlinear solvers agree with each
other to :math:`10^{-6}` in every nodal velocity, a stronger statement than
either comparison with the published profile, and a final test confirms that the divergence of
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

**The advection scheme.**  At a high cell Péclet number
:math:`\mathrm{Pe}_h = |\mathbf{v}| h / (2 \kappa)`, with
:math:`\kappa = k / (\rho c_p)`, the central source :eq:`heatconvection`
oscillates.  The heat that a ``nonisothermal_flow`` coupling carries takes, by
default, the streamline stabilisation below (SUPG), which measured the closest
on boundary layers and jumps.  ``advection_scheme`` of ``heat_transfer``
chooses another one (``"central"``, or with DMCDM the face fluxes
``"upwind"``, ``"linear_upwind"`` and ``"limited"`` of :doc:`finite_volume`),
because the best scheme depends
on the case: where conduction matters as much as convection, as in resolved
natural convection, central can be closer (below).  The face schemes write the
term with the faces of each control volume, as :math:`\rho c_p \sum_f m\, T_f`,
with the flux :math:`m` of the velocity that the mass equation conserves, so
that the heat is conserved exactly and the temperature does not depend on its
scale.  The inertia of a flow takes the streamline stabilisation
``momentum_stabilization``, with no option.

``"SUPG"`` adds ``heat_streamline_stabilization``,
the streamline upwind Petrov-Galerkin term of [BrooksHughes1982]_.  It is the
flux :math:`\mathbf{F} = \tau\, \mathbf{v}\, r`, which adds
:math:`\int_\Omega \tau\, (\mathbf{v} \cdot \nabla w)\, r \, d\Omega` to the
equation of the test function :math:`w`, with the residual

.. math::

   r = \rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T
   \right) - q .

The kernel builds :math:`r` as the sum of the sources of every other kernel of
the temperature: the convection, the heat capacity term and each heat source,
also one added with ``add_kernel``.  A source missing from :math:`r` would make
the stabilisation act against it: with the heat source added as a kernel and
only the source of the physics in :math:`r`, the order on a smooth solution
fell from 2.8 to 1.2.  The parameter is that of [Tezduyar1991]_

.. math::

   \tau = \left[ \left( \frac{2 |\mathbf{v}|}{h} \right)^2
   + 9 \left( \frac{4 \kappa}{h^2} \right)^2
   + \left( \frac{2}{\Delta t} \right)^2 \right]^{-1/2} ,

with the factor 9 of the flow (above).  On the boundary layer of
:math:`\mathrm{Pe} = 100` on 10, 20 and 40 elements the maximum error is 0.028,
0.013 and 0.0024 with it, and 0.083, 0.134 and 0.143 without it, which
overdiffuses where conduction and convection are comparable.

The residual leaves out the conduction term :math:`-\nabla \cdot (k \nabla T)`,
whose second derivatives vanish inside a linear element.  For the exact
solution this term is not zero, so the stabilisation adds an error of the order
of :math:`\tau k |\Delta T|`.  It is small where convection dominates, where
:math:`\tau \approx h / (2 |\mathbf{v}|)`, and zero for a harmonic temperature.
On :math:`T = \sin(\pi x) \sin(\pi y)` with :math:`k = 0.01` the error of SUPG
is :math:`1.4 \times 10^{-2}`, :math:`2.3 \times 10^{-3}` and
:math:`2.6 \times 10^{-4}` on 8, 16 and 32 elements per side (Galerkin's
:math:`2.6 \times 10^{-2}`, :math:`6.3 \times 10^{-3}` and
:math:`1.6 \times 10^{-3}`).  On the harmonic :math:`T = e^x \sin y` both converge at second
order.  For the inertia of a flow, SUPG is ``momentum_stabilization``, which
``fluid_dynamics`` adds with both methods.

On the skew jump of [BrooksHughes1982]_ with the outflow held at zero (a
:math:`24 \times 24` mesh, :math:`k = 10^{-6}`), Galerkin's temperature reaches
409, and that of SUPG stays in :math:`[-0.12, 1.08]` away from the outflow
layer, with 1.6 at the node next to the outflow corner, where the layer meets
the jump.  This overshoot at a layer is a known property of SUPG, which is not
bounded.  The bounded schemes are those of the control volume methods.

**The time derivative inside a residual.**  The theta method weights a steady
kernel by :math:`\theta` at the new time and by :math:`1 - \theta` at the old
time, where the kernel has no old value and its time derivative is zero.  A
residual-based stabilisation puts the time derivative inside a steady kernel,
so the derivative would enter the step with the weight :math:`\theta`.  The
context of the kernel therefore carries the inverse of the kernel's own weight
for it: :math:`1/\theta` at the new time for this kernel, 1 for the
stabilisation of a flow, which the step takes at the new time (see
"Transient flow"), and 0 at the old time.  The derivative enters with the
weight 1.  The explicit step (:math:`\theta = 0`)
cannot give it a weight, so it refuses these kernels.  With Crank-Nicolson, SUPG then converges
at second order on :math:`T = e^x \sin y\, (1 + t)`, with the error of backward
Euler.  With the weight :math:`\theta` it converged at order 0.7 to 1.1.

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
   :math:`10^3`  1.117       1.118     3.659        3.649     3.710        3.697
   :math:`10^4`  2.242       2.243     16.27        16.18     19.72        19.62
   :math:`10^5`  4.507       4.519     35.09        34.73     68.92        68.59
   :math:`10^6`  8.774       8.800     65.78        64.63     224.6        219.4
   ============  ==========  ========  ===========  ========  ===========  ========

Here :math:`u_m` is the largest horizontal velocity on the vertical centreline
and :math:`v_m` the largest vertical velocity on the horizontal centreline.  The
Nusselt numbers agree to within 0.3 % and the velocity maxima to within 1.8 %,
except :math:`v_m` at :math:`Ra = 10^6`, which is 2.4 % high, where the wall jet
is thinnest.  On a :math:`64 \times 64` mesh this value is 221.4.  The heat
carries the default streamline stabilisation, whose residual on these linear
elements leaves the conduction term out.  That costs a little here, where conduction matters as
much as convection: with ``advection_scheme="central"`` the differences were
0.2 % and 0.6 % (1.8 % for :math:`v_m` at :math:`Ra = 10^6`).  The finite
element method gives Nusselt numbers within 0.2 % of the same values.  The heat entering at the hot wall leaves at the cold
wall to round-off, and on a :math:`16 \times 16` mesh at :math:`Ra = 10^4` the
relative Newton steps of the last load step fall as
:math:`1.8 \times 10^{-3}`, :math:`1.5 \times 10^{-6}`,
:math:`2.1 \times 10^{-12}`, which is quadratic convergence.  The tests are in
``tests/python/test_multiphysics.py`` and the example script is
``examples/natural_convection.py``.
