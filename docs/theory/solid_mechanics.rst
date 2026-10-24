Linear elasticity
=================

This chapter describes the small-strain, linearly elastic solid mechanics of
the ``solid_mechanics`` module: the equilibrium equations that are solved, the
kinematics and the constitutive law, the four idealisations supported (plane
stress, plane strain, axisymmetric and three-dimensional), thermal strain, the boundary conditions belonging to the elasticity
duality pair, and the out-of-plane thickness.  It assumes undergraduate solid
mechanics but no acquaintance with this library.  Throughout, a stress
component is positive in tension, and a displacement, traction or force
component is positive when it points along the positive direction of its own
coordinate axis.

The equilibrium equations
-------------------------

Consider a body occupying a region :math:`\Omega` and carrying a Cauchy stress
field :math:`\sigma_{ij}(\mathbf{x})`, in which the first index names the face
the stress acts on and the second the direction it acts along.  Requiring
every part of the body to be in static equilibrium, and letting that part
shrink to a point, gives the local balance of linear momentum

.. math::
   :label: elasticity-equilibrium

   \frac{\partial \sigma_{ij}}{\partial x_j} + b_i = 0 \qquad\text{in } \Omega ,

where :math:`b_i` is the body force per unit volume along the :math:`i`
direction, such as the weight :math:`-\rho g` of the material itself, and
repeated indices are summed.  In vector form this is
:math:`-\nabla\cdot\boldsymbol{\sigma} = \mathbf{b}`, the divergence being
taken row by row.  Balance of angular momentum makes the stress tensor
symmetric, :math:`\sigma_{ij} = \sigma_{ji}`, which is why the module stores
only the six independent components of the stress.

Equation :eq:`elasticity-equilibrium` comprises one equation per displacement
component, and the library treats it as such.  Every equation
in the framework is written in the canonical conservation form

.. math::
   :label: elasticity-canonical

   -\nabla\cdot\mathbf{F}_i + S_i = 0 ,

with a flux vector :math:`\mathbf{F}_i` and a source :math:`S_i` that may both
depend on all the unknowns and their gradients.  Elasticity fits this form by
taking, for the equation of the displacement component :math:`u_i`,

.. math::
   :label: elasticity-flux

   \mathbf{F}_i = h\,(\sigma_{i1},\, \sigma_{i2},\, \sigma_{i3}) ,
   \qquad S_i = -h\,b_i ,

so that the flux of the :math:`i` equation is the :math:`i` th row of the
stress tensor and its source is minus the corresponding body force component.
The factor :math:`h` is the out-of-plane thickness treated at the end of this
chapter.  Substituting :eq:`elasticity-flux` into :eq:`elasticity-canonical`
and dividing by :math:`h` recovers :eq:`elasticity-equilibrium`, which fixes
the sign convention: a positive body force component drives the corresponding
displacement in the positive coordinate direction.  The natural, or secondary,
quantity of a canonical equation is the outward normal component of its flux,
here :math:`\mathbf{n}\cdot\mathbf{F}_i = h\,n_j\sigma_{ij} = h\,t_i`, the
:math:`i` th component of the surface traction times the thickness, so the
traction component is the natural partner of the displacement component with
no further construction.

The module splits the work between two kinds of object.  The **property object**,
``small_strain_stress``, is independent of the equilibrium statement: given
the displacement gradients at an integration point it computes the strain and
the stress and provides them as the properties ``stress``, ``strain``,
``elastic_strain`` and ``volumetric_strain``.  The **kernel**, ``stress_divergence``, is independent
of the constitutive law: it receives the index of the component :math:`i` it
represents, reads the stress property, and returns the flux
:eq:`elasticity-flux` together with the axisymmetric source described below.
One kernel instance is added per displacement variable, with ``component`` set
to ``0`` for :math:`x` or :math:`r`, ``1`` for :math:`y` or :math:`z`, and
``2`` for :math:`z` in three dimensions.  That index must agree with the
component the kernel's ``variable`` actually represents.  The agreement is not
checked, and a mismatch silently assembles the wrong row of the stress.

The separation has three advantages.  First, one kernel serves all four
formulations and both the dual mesh control domain method and the finite
element method, because equilibrium is the same statement in every case and
only the stress differs.  Second, a new constitutive model needs only a
property object that declares a ``stress`` property in the same layout, with no change
to the equilibrium code.  Third, because property objects and kernels are written in
the forward-mode automatic differentiation type used throughout the framework,
the
derivative of the stress with respect to every degree of freedom is carried
through the constitutive law, so the Newton Jacobian stays exact even when the
stress depends on another field such as the temperature.

The ``solid_mechanics`` physics assembles the whole set: it creates the
displacement variables, adds one property object and one kernel per component, and,
when a ``body_force`` is given, adds a ``body_force`` kernel per component
whose intensity has already been multiplied by the thickness.

Kinematics and the constitutive law
-----------------------------------

Strain measure and Voigt ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deformation is measured by the infinitesimal strain tensor, the symmetric part
of the displacement gradient,

.. math::
   :label: elasticity-strain

   \varepsilon_{ij} = \tfrac{1}{2}
   \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) ,

which is the appropriate measure when displacements and rotations are small
enough that the difference between deformed and undeformed geometry can be
ignored.  Equilibrium is then written on the undeformed body.  Symmetric
tensors are stored as six-component arrays in the Voigt ordering

.. math::

   (\,0,\ 1,\ 2,\ 3,\ 4,\ 5\,) \;\longleftrightarrow\;
   (\,xx,\ yy,\ zz,\ yz,\ xz,\ xy\,) ,

where slot 2 holds the hoop component :math:`\theta\theta` in the axisymmetric
case and slot 5 the :math:`rz` shear there.  This is the ordering the
``stress`` and ``strain`` properties present to post-processing.  The three
shear slots hold **engineering** shear strains, twice the tensor components,

.. math::

   \gamma_{xy} = 2\varepsilon_{xy}
   = \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} ,
   \quad
   \gamma_{yz} = \frac{\partial v}{\partial z} + \frac{\partial w}{\partial y} ,
   \quad
   \gamma_{xz} = \frac{\partial u}{\partial z} + \frac{\partial w}{\partial x} ,

with :math:`(u, v, w)` the variables named in ``displacements``.  The
distinction matters because the shear stiffness that relates a shear stress to
an engineering shear strain is the shear modulus :math:`G` itself, whereas the
stiffness that relates it to a tensor shear strain is :math:`2G`.  The
property ``volumetric_strain`` holds
:math:`\varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz}`, the sum of the
normal slots, which is the relative change of volume.

The constitutive law is a matrix product in the same ordering,

.. math::
   :label: elasticity-constitutive

   \sigma_I = \sum_{J=0}^{5} C_{IJ}\, \varepsilon_J ,

with :math:`C_{IJ}` the isotropic stiffness of the formulation.  The
stiffness is built from Young's modulus :math:`E`, a stiffness in
force per unit area, and Poisson's ratio :math:`\nu`, the ratio of transverse
contraction to axial extension, which must satisfy :math:`-1 < \nu < 1/2`.  It
is convenient to name the Lamé constants

.. math::

   \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)} , \qquad
   \mu = G = \frac{E}{2(1+\nu)} ,

in which :math:`\mu` is the shear modulus.  Both :math:`\lambda` and the
plane-strain, axisymmetric and three-dimensional stiffnesses below become
singular as :math:`\nu \to 1/2`, so a nearly incompressible material needs
care.  Young's modulus has no default.  When Poisson's ratio is not given, it is
the property ``poissons_ratio`` if a property object declares it, and otherwise
:math:`1/3`, the default of the library, which the report marks as a default.
Which stiffness is built is chosen by ``formulation``, one of
``plane_stress``, ``plane_strain``, ``axisymmetric``, ``axisymmetric_1d`` or
``three_dimensional``.  Any other string is an error.  The number of names
in ``displacements`` and the dimension of the mesh must match the choice: two
of each for the plane and axisymmetric formulations, three for the
three-dimensional one.

Plane stress
^^^^^^^^^^^^

Plane stress models a thin plate loaded in its own plane, thin enough that the
free faces force :math:`\sigma_{zz} = \sigma_{yz} = \sigma_{xz} = 0` through
the thickness.  The material contracts freely through the thickness, so
:math:`\varepsilon_{zz}` is in general non-zero, but it does no work and is
eliminated,
leaving

.. math::

   \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix}
   = \frac{E}{1-\nu^{2}}
   \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\
     0 & 0 & \tfrac{1}{2}(1-\nu) \end{bmatrix}
   \begin{Bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{Bmatrix} ,

the shear entry being :math:`E/[2(1+\nu)] = G`.  The ``stress`` property
reports :math:`\sigma_{zz} = 0`, which is what the idealisation asserts, and
the ``strain`` property reports the out-of-plane strain that keeps it so,

.. math::

   \varepsilon_{zz} = -\frac{\nu}{1-\nu}\,(\varepsilon_{xx} + \varepsilon_{yy}),

of the elastic strains, plus the eigenstrains.  The ``volumetric_strain``
includes it.  A two-dimensional Cartesian problem names its formulation,
``plane_stress`` or ``plane_strain``, because neither is right for every body.

Plane strain
^^^^^^^^^^^^

Plane strain models the opposite extreme, a body so long in :math:`z`, or so
restrained there, that :math:`\varepsilon_{zz} = \gamma_{yz} = \gamma_{xz} =
0`: a long dam, a tunnel lining, a thick-walled pipe held between rigid ends.
With :math:`f = E / [(1+\nu)(1-2\nu)]` the in-plane law is

.. math::

   \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix}
   = \begin{bmatrix} f(1-\nu) & f\nu & 0 \\ f\nu & f(1-\nu) & 0 \\
     0 & 0 & \tfrac{E}{2(1+\nu)} \end{bmatrix}
   \begin{Bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{Bmatrix} .

Because the out-of-plane strain vanishes while the out-of-plane stress does
not, the property object also fills the :math:`zz` row with :math:`C_{20} = C_{21} =
f\nu` and :math:`C_{22} = f(1-\nu)`.  With :math:`\varepsilon_{zz} = 0` that
row returns :math:`\sigma_{zz} = f\nu\,(\varepsilon_{xx} + \varepsilon_{yy}) =
\nu\,(\sigma_{xx} + \sigma_{yy})`, post-computed for reporting.  It plays no
part in the two in-plane equations, because the :math:`zz` column of the
in-plane rows is zero.

Axisymmetric
^^^^^^^^^^^^

The axisymmetric formulation models a body of revolution loaded so that
nothing depends on the circumferential coordinate :math:`\theta` and there is
no circumferential displacement.  The mesh is a cut through the body in the
:math:`(r, z)` half-plane with :math:`r = x_1 \ge 0`, and the two variables
are the radial displacement :math:`u_r` and the axial displacement
:math:`u_z`.  The problem must be created with ``coordinates="axisymmetric"``,
which is what makes every control-domain integral carry the measure
:math:`2\pi r`.

Three strains are the familiar ones, :math:`\varepsilon_{rr} = \partial
u_r/\partial r`, :math:`\varepsilon_{zz} = \partial u_z/\partial z` and
:math:`\gamma_{rz} = \partial u_r/\partial z + \partial u_z/\partial r`, but
there is a fourth that arises purely from the geometry.  A material circle of
radius :math:`r` about the axis has circumference :math:`2\pi r`.  After
deformation its radius is :math:`r + u_r`, so its relative change of length is

.. math::
   :label: elasticity-hoop-strain

   \varepsilon_{\theta\theta}
   = \frac{2\pi (r + u_r) - 2\pi r}{2\pi r} = \frac{u_r}{r} .

A radial displacement therefore stretches the material in the hoop direction
even where its radial gradient vanishes, which is why a pressurised cylinder
carries hoop stress.  The hoop strain occupies slot 2 and the :math:`rz` shear
slot 5, and the stiffness is that of a three-dimensional isotropic material
for the three normal components together with one shear term,

.. math::

   C_{IJ} = \lambda + 2\mu\,\delta_{IJ}
   \quad (I, J \in \{rr,\, zz,\, \theta\theta\}) , \qquad C_{55} = \mu ,

so that :math:`\sigma_{rr} = (\lambda + 2\mu)\varepsilon_{rr} +
\lambda\varepsilon_{zz} + \lambda\varepsilon_{\theta\theta}` and
:math:`\sigma_{rz} = \mu\gamma_{rz}`, for example.  The remaining shear slots
stay at zero because axisymmetry forbids :math:`r\theta` and :math:`z\theta`
shear.

The hoop stress enters equilibrium as a source term, for the following
reason.  Radial equilibrium in cylindrical coordinates,
:math:`\partial\sigma_{rr}/\partial r + \partial\sigma_{rz}/\partial z +
(\sigma_{rr} - \sigma_{\theta\theta})/r + b_r = 0`, regroups as

.. math::
   :label: elasticity-axisymmetric-radial

   \frac{1}{r}\frac{\partial (r\,\sigma_{rr})}{\partial r}
   + \frac{\partial \sigma_{rz}}{\partial z}
   - \frac{\sigma_{\theta\theta}}{r} + b_r = 0 .

The first two terms are exactly the divergence of the in-plane vector
:math:`(\sigma_{rr}, \sigma_{rz})` taken with the measure :math:`2\pi r`, so
they are supplied by the flux :eq:`elasticity-flux` unchanged: the factor
:math:`r` is already in the integration measure the coordinate system
provides.  The hoop term is different.  Since nothing varies with
:math:`\theta`, the hoop stress is differentiated with respect to a direction
the mesh does not resolve, and its contribution to the radial balance remains
as an algebraic term that cannot be written as the divergence of a vector on
the :math:`(r, z)` mesh.  Comparison with :eq:`elasticity-canonical` shows that
it must be carried as the source :math:`S_r = \sigma_{\theta\theta}/r`, which
``stress_divergence`` adds to the radial equation and to that equation alone.
The axial equation has no such term.  Setting
``axisymmetric_hoop_term=False`` removes it, which isolates its effect but
leaves a body with no hoop stiffness.  The resulting model does not represent
a physical axisymmetric body.

On the axis :math:`r = 0` both :eq:`elasticity-hoop-strain` and the source are
indeterminate, and each is handled explicitly.  A regular solution has
:math:`u_r(0) = 0`, so the limit of :math:`u_r/r` as :math:`r \to 0` is
:math:`\partial u_r/\partial r`, and that derivative is what the property object uses
whenever an integration point falls exactly on the axis.  There the hoop
strain and the radial strain coincide.  The kernel returns a zero source at
:math:`r = 0`, which is consistent because the measure :math:`2\pi r` of the
surrounding control domain vanishes there as well.  Neither treatment supplies
the symmetry condition itself: a mesh reaching the axis still needs
:math:`u_r = 0` prescribed there, as a boundary condition like any other.

Three dimensions
^^^^^^^^^^^^^^^^

The three-dimensional formulation makes no assumption about the geometry and
uses the full isotropic stiffness,

.. math::

   C_{IJ} = \lambda + 2\mu\,\delta_{IJ} \quad (I, J \in \{0, 1, 2\}) ,
   \qquad C_{33} = C_{44} = C_{55} = \mu ,

which is :math:`\sigma_{ij} = \lambda\,\varepsilon_{kk}\,\delta_{ij} +
2\mu\,\varepsilon_{ij}` in Voigt form.  The shear rows carry :math:`\mu`,
where the tensor form has :math:`2\mu`, because the shear slots hold
engineering strains.
Three displacement variables and a three-dimensional mesh are required, and
the thickness must be left at one.

Thermal strain
--------------

A temperature change makes an unconstrained body expand without producing
stress.  Stress arises only from the part of the strain not accounted for by
that free expansion.  With :math:`\alpha` the coefficient of thermal
expansion per degree, :math:`T` the temperature field and
:math:`T_{\mathrm{ref}}` the temperature at which the body is free of stress,
the thermal strain is :math:`\varepsilon^{\mathrm{th}} = \alpha\,(T -
T_{\mathrm{ref}})`.  It is an *eigenstrain*, a strain that the material takes
without stress: ``thermal_expansion_eigenstrain`` declares it, and
``small_strain_stress`` subtracts the eigenstrains it lists in
``eigenstrain_names`` from the strain before :eq:`elasticity-constitutive`,

.. math::
   :label: elasticity-thermal

   \sigma_I = \sum_{J=0}^{5} C_{IJ}
              \left( \varepsilon_J - \varepsilon^{\mathrm{th}}_J \right) ,
   \qquad \varepsilon^{\mathrm{th}}_J = \alpha\,(T - T_{\mathrm{ref}})
   \ (J = 0, 1, 2), \quad \varepsilon^{\mathrm{th}}_J = 0 \ (J = 3, 4, 5).

The thermal part is subtracted from the three normal strain components only,
because an isotropic expansion changes lengths equally in all directions and
produces no shear.  Heating a fully restrained body above
:math:`T_{\mathrm{ref}}` makes :math:`\varepsilon_J - \varepsilon^{\mathrm{th}}_J`
negative and puts the body into compression.  In plane strain the
out-of-plane strain is zero, so the thermal strain loads the :math:`zz`
direction too, and a body restrained in its plane carries
:math:`\sigma_{xx} = \sigma_{yy} = -E\,\alpha\,(T - T_{\mathrm{ref}})/(1 - 2\nu)`.
In plane stress the body expands freely through its thickness.  The
``stress_free_temperature`` :math:`T_{\mathrm{ref}}` has no default, because
no value is right for every problem.  The ``thermal_expansion`` coupling of
the physics adds the eigenstrain and lists it.

The ``temperature`` parameter of ``thermal_expansion_eigenstrain`` names an ordinary variable of the problem, and
this has a useful consequence.  If that variable is itself being solved for,
by a heat conduction kernel elsewhere in the same problem, the thermal stress
depends on an unknown of the system.  Because the property object is written in the
automatic differentiation type, the derivative of the stress with respect to
the temperature degrees of freedom is carried through :eq:`elasticity-thermal`
alongside the derivatives with respect to the displacements.  The
thermo-mechanical coupling is therefore differentiated exactly and Newton's
method converges at its usual rate on the fully coupled system, with nothing
lagged or iterated by hand.  A prescribed temperature
field can be supplied as a variable fixed by a Dirichlet condition, in which
case the coupling contributes nothing to the Jacobian.

Boundary conditions
-------------------

Every canonical equation has a duality pair: the primary variable, the unknown
itself, and the secondary variable, the outward normal component of the flux.
For the equation of the displacement component :math:`u_i` the pair is
:math:`u_i` and the traction component :math:`t_i = n_j\sigma_{ij}`, as shown
above.  At every boundary point exactly one member of the pair is specified,
and the choice is made independently for each component: an edge may have its
normal displacement prescribed and its tangential traction prescribed, which
is how a frictionless symmetry plane is modelled.  A boundary carrying no
condition at all for a component has the natural condition with a zero value,
:math:`t_i = 0`, so a traction-free surface needs no input.

Prescribed displacement
^^^^^^^^^^^^^^^^^^^^^^^

``Dirichlet`` prescribes the primary variable,
:math:`u_i = g(\mathbf{x}, t)` at every node of the named boundary, replacing
that node's equilibrium equation.  A zero value is a rigid support in that
direction and a non-zero value a prescribed settlement or stretch.  Prescribing
one component and leaving the other free is a roller support, and prescribing
all of them is a clamp.  By default a prescribed displacement is held fixed
while loads are ramped through load steps, and ``scale_with_load=True`` makes
it ramp with them, as a displacement-controlled test requires.

Prescribed traction
^^^^^^^^^^^^^^^^^^^

``traction`` prescribes the secondary variable directly:
the component of the surface traction, in force per unit area, conjugate to
the variable the object is attached to.  Because that variable determines the
component, the object has no ``component`` parameter and is added once per
displacement variable on the loaded boundary.  The value may be a constant or
a function of position and time.  The sign convention is tied to the
coordinate axes, independently of the orientation of the surface, so a
positive value acts along the positive direction of that variable's axis
whichever way the outward normal points, and one positive value pulls on a
right-hand edge while pushing on a left-hand edge.  The value
is multiplied by the thickness before it enters the residual.

Pressure
^^^^^^^^

``pressure`` applies a load that follows the orientation of
the surface.  A fluid pressure :math:`p` acts normal to the surface and
towards the body, so the traction it exerts is

.. math::
   :label: elasticity-pressure

   t_i = -p\, n_i ,

with :math:`\mathbf{n}` the outward unit normal.  The convention is contained
entirely in the minus sign: a positive pressure presses inward, against the outward
normal, and a negative value is a suction.  Because each component of the
normal multiplies the pressure separately, one instance acts on each
displacement variable.  The object takes the list ``displacements`` and finds
the component of each instance from the place of its variable in that list, so
the component cannot disagree with the variable.  The pressure is multiplied by
the thickness in plane problems.

Concentrated force
^^^^^^^^^^^^^^^^^^

``point_source`` applies a concentrated force at one or more nodes, either by
naming a boundary whose nodes all receive the load or by giving coordinates
that are assigned to the nearest node.  The magnitude is a force, in the units
conjugate to the displacement, and a positive value acts along the positive
direction of that variable's axis.  A concentrated force is an idealisation
that produces an unbounded stress at the loaded point in the continuum
solution, so the stress reported in the neighbouring elements depends on the
mesh and should not be read as a converged value, although displacements away
from the load converge normally.

Reactions at a prescribed-displacement boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a node's equation is replaced by a prescribed displacement, the equation
that was replaced is not discarded.  Evaluated at the converged solution it
returns the secondary variable of that equation integrated over the part of
the node's control domain lying on the boundary,

.. math::

   Q_I = \int_{\partial \Omega_I \cap \partial\Omega} h\, t_i \,\mathrm{d}S ,

which is the force the support transmits to the body at that node, positive
along the positive direction of the variable's axis.  This is what
:meth:`dualmesh.Problem.reactions` returns node by node and the quantity to
compare against a hand-calculated support reaction.  Summing over a boundary
with :meth:`dualmesh.Problem.total_reaction` gives the resultant support force
there, which for a body in equilibrium balances the applied loading.

The following error is easy to make, and its result appears plausible.  The
reaction stored at a node is the whole of the support force at
that node, gathered from every part of the boundary touching its control
domain.  A node at the corner where two constrained boundaries meet therefore
belongs to both and appears in the list returned for each, so adding the
totals of two boundaries that share such a node counts that node's reaction
twice and overstates the resultant.  When a resultant over several boundaries
is wanted, the node lists should be merged and each node counted once, or a
single boundary holding all the constrained nodes should be defined in the
mesh.

The thickness parameter
-----------------------

The ``thickness`` parameter :math:`h` is the out-of-plane dimension of a plane
problem: the thickness of the sheet in plane stress, or the length of the
slice being modelled in plane strain.  It multiplies the entire equilibrium
equation, i.e., the flux :eq:`elasticity-flux` assembled by
``stress_divergence``, the traction assembled by
``traction`` and the pressure assembled by
``pressure``, converting stresses per unit area into forces
per unit length of the cross-section so that every residual has the
dimensions of a force.  It applies to ``plane_stress`` and ``plane_strain``
only: in three dimensions there is no out-of-plane dimension to account for,
and in the axisymmetric case the measure :math:`2\pi r` already supplies the
circumferential extent of the material surrounding each control domain, so in
both cases the value stays at its default of one, and another value is
refused.

The thickness is given once, to the ``stress_divergence`` kernels.  The
``traction``, ``pressure`` and ``gap_contact`` conditions of a displacement
read it from the kernel of that displacement, so the loads and the stiffness
are always scaled by the same :math:`h`, which cancels out of the displacement
solution of a linear problem.  A condition reads one thickness: when the
kernels of one displacement have different thicknesses (blocks of different
thickness), the condition is refused.  The ``solid_mechanics`` physics passes
its ``thickness`` to its kernels and its body force.

One load is deliberately excluded from this rule.  A ``point_source`` is
applied **without** the thickness factor, because a concentrated load is a
total force, whereas the traction and the pressure are forces per unit area.
The value supplied must therefore already be the total force acting through
the whole thickness: a line load of :math:`q` per unit length along the
out-of-plane direction becomes a point force of :math:`q h`.

Verification
------------

The module is verified in ``tests/python/test_solid_mechanics.py``.  A
quadrant of a plate under uniform edge stress has the exact solution
:math:`u(a) = t_0 a / E`, which the computed edge displacement matches to
machine precision with an exactly uniform stress field.  A cantilevered plate
under a uniform edge load reproduces the published dual mesh values
[Reddy2024]_ on meshes of one, two, four and eight elements a side, both for
the corner displacements and for the stress at the centre of the element
nearest the support.  A thick-walled cylinder under internal pressure, solved
as a plane-strain quadrant of an annulus with symmetry conditions on the
straight edges and ``pressure`` on the bore, reproduces the
published quadrilateral values to about 0.5 %, which is the accuracy the
published description of the mesh permits, converges monotonically to the
Lamé solution on quadrilateral and triangular meshes alike, and agrees to
within 0.2 % with the same cylinder solved as an axisymmetric problem.
