Linear elasticity
=================

This chapter describes the small-strain, linearly elastic solid mechanics of
the ``solid_mechanics`` module: the equilibrium equations that are solved, the
kinematics and the constitutive law, the four idealisations supported (plane
stress, plane strain, axisymmetric and three-dimensional), orthotropic
materials, thermal strain, the boundary conditions belonging to the elasticity
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
six independent components rather than nine.

Equation :eq:`elasticity-equilibrium` is not one equation but one per
displacement component, and that is how the library treats it.  Every equation
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
stress tensor and its source is minus the corresponding body force component;
the factor :math:`h` is the out-of-plane thickness treated at the end of this
chapter.  Substituting :eq:`elasticity-flux` into :eq:`elasticity-canonical`
and dividing by :math:`h` recovers :eq:`elasticity-equilibrium`, which fixes
the sign convention: a positive body force component drives the corresponding
displacement in the positive coordinate direction.  The natural, or secondary,
quantity of a canonical equation is the outward normal component of its flux,
here :math:`\mathbf{n}\cdot\mathbf{F}_i = h\,n_j\sigma_{ij} = h\,t_i`, the
:math:`i` th component of the surface traction times the thickness, so the
traction component is the natural partner of the displacement component with
no further construction.

The module splits the work between two kinds of object.  The **material**,
``LinearElasticStress``, knows nothing about equilibrium: given the
displacement gradients at an integration point it computes the strain and the
stress and publishes them as the properties ``stress``, ``strain`` and
``volumetric_strain``.  The **kernel**, ``StressDivergence``, knows nothing
about the constitutive law: it is told which component :math:`i` it
represents, reads the stress property, and returns the flux
:eq:`elasticity-flux` together with the axisymmetric source described below.
One kernel instance is added per displacement variable, with ``component`` set
to ``0`` for :math:`x` or :math:`r`, ``1`` for :math:`y` or :math:`z`, and
``2`` for :math:`z` in three dimensions.  That index must agree with the
component the kernel's ``variable`` actually represents; the agreement is not
checked, and a mismatch assembles the wrong row of the stress silently.

The separation earns its keep three times over.  One kernel serves all four
formulations and both the dual mesh control domain method and the finite
element method, because equilibrium is the same statement in every case and
only the stress differs.  A new constitutive model needs only a material that
declares a ``stress`` property in the same layout, with no change to the
equilibrium code.  And because materials and kernels are written in the
forward-mode automatic differentiation type used throughout the framework, the
derivative of the stress with respect to every degree of freedom is carried
through the constitutive law, so the Newton Jacobian stays exact even when the
stress depends on another field such as the temperature.

In Python the whole set is assembled by
:func:`dualmesh.physics.add_plane_elasticity`, which creates the displacement
variables if needed, adds one material and one kernel per component, and, when
a ``body_force`` is given, adds a ``BodyForce`` kernel per component whose
intensity has already been multiplied by the thickness.

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
ignored; equilibrium is then written on the undeformed body.  Symmetric
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
distinction matters because the shear stiffness is then the shear modulus
itself rather than twice it.  The property ``volumetric_strain`` holds
:math:`\varepsilon_{xx} + \varepsilon_{yy} + \varepsilon_{zz}`, the sum of the
normal slots, which is the relative change of volume.

The constitutive law is a matrix product in the same ordering,

.. math::
   :label: elasticity-constitutive

   \sigma_I = \sum_{J=0}^{5} C_{IJ}\, \varepsilon_J ,

with :math:`C_{IJ}` assembled once when the material is constructed.  The
isotropic stiffness is built from Young's modulus :math:`E`, a stiffness in
force per unit area, and Poisson's ratio :math:`\nu`, the ratio of transverse
contraction to axial extension, which must satisfy :math:`-1 < \nu < 1/2`.  It
is convenient to name the Lamé constants

.. math::

   \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)} , \qquad
   \mu = G = \frac{E}{2(1+\nu)} ,

in which :math:`\mu` is the shear modulus.  Both :math:`\lambda` and the
plane-strain, axisymmetric and three-dimensional stiffnesses below become
singular as :math:`\nu \to 1/2`, so a nearly incompressible material needs
care, and the default :math:`\nu = 0`, which gives no transverse coupling at
all, is rarely the material intended.  Which stiffness is built is chosen by
``formulation``, one of ``plane_stress``, ``plane_strain``, ``axisymmetric``
or ``three_dimensional``; any other string is an error.  The number of names
in ``displacements`` and the dimension of the mesh must match the choice: two
of each for the plane and axisymmetric formulations, three for the
three-dimensional one.

Plane stress
^^^^^^^^^^^^

Plane stress models a thin plate loaded in its own plane, thin enough that the
free faces force :math:`\sigma_{zz} = \sigma_{yz} = \sigma_{xz} = 0` through
the thickness.  The material contracts freely through the thickness, so
:math:`\varepsilon_{zz}` is not zero, but it does no work and is eliminated,
leaving

.. math::

   \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix}
   = \frac{E}{1-\nu^{2}}
   \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\
     0 & 0 & \tfrac{1}{2}(1-\nu) \end{bmatrix}
   \begin{Bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{Bmatrix} ,

the shear entry being :math:`E/[2(1+\nu)] = G`.  The row that would give
:math:`\sigma_{zz}` is left at zero, so the ``stress`` property reports
:math:`\sigma_{zz} = 0`, which is what the idealisation asserts.  This is the
default formulation.

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
not, the material also fills the :math:`zz` row with :math:`C_{20} = C_{21} =
f\nu` and :math:`C_{22} = f(1-\nu)`.  With :math:`\varepsilon_{zz} = 0` that
row returns :math:`\sigma_{zz} = f\nu\,(\varepsilon_{xx} + \varepsilon_{yy}) =
\nu\,(\sigma_{xx} + \sigma_{yy})`, post-computed for reporting; it plays no
part in the two in-plane equations, the :math:`zz` column of the in-plane rows
being zero.

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
radius :math:`r` about the axis has circumference :math:`2\pi r`; after
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

The hoop stress does not enter equilibrium as part of a flux, and the reason
is worth spelling out.  Radial equilibrium in cylindrical coordinates,
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
the mesh does not resolve, and what it contributes to the radial balance
survives as an algebraic term rather than as the divergence of anything on the
:math:`(r, z)` mesh.  Comparison with :eq:`elasticity-canonical` shows that it
must be carried as the source :math:`S_r = \sigma_{\theta\theta}/r`, which
``StressDivergence`` adds to the radial equation and to that equation alone;
the axial equation has no such term.  Setting
``axisymmetric_hoop_term=False`` removes it, which isolates its effect but
leaves a body with no hoop stiffness and is not the intended model.

On the axis :math:`r = 0` both :eq:`elasticity-hoop-strain` and the source are
indeterminate, and each is handled explicitly.  A regular solution has
:math:`u_r(0) = 0`, so the limit of :math:`u_r/r` as :math:`r \to 0` is
:math:`\partial u_r/\partial r`, and that derivative is what the material uses
whenever an integration point falls exactly on the axis; there the hoop strain
and the radial strain coincide.  The kernel returns a zero source at
:math:`r = 0`, which is consistent because the measure :math:`2\pi r` of the
surrounding control domain vanishes there as well.  Neither device supplies
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
2\mu\,\varepsilon_{ij}` in Voigt form; the shear rows carry :math:`\mu` rather
than :math:`2\mu` precisely because the shear slots hold engineering strains.
Three displacement variables and a three-dimensional mesh are required, and
the thickness must be left at one.

Orthotropic materials
---------------------

A material whose stiffness differs along two perpendicular in-plane
directions, such as a unidirectional composite lamina or a rolled sheet, is
described in a plane problem by four reduced stiffnesses, given through
``stiffness_c11``, ``stiffness_c12``, ``stiffness_c22`` and ``stiffness_c66``:

.. math::

   \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix}
   = \begin{bmatrix} c_{11} & c_{12} & 0 \\ c_{12} & c_{22} & 0 \\
     0 & 0 & c_{66} \end{bmatrix}
   \begin{Bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{Bmatrix} .

Here :math:`c_{11}` and :math:`c_{22}` are the direct stiffnesses along the
two material axes, :math:`c_{12}` is the coupling term, placed symmetrically
so that the stiffness stays symmetric, and :math:`c_{66}` is the in-plane
shear stiffness relating :math:`\sigma_{xy}` to the engineering shear strain
:math:`\gamma_{xy}` rather than to :math:`\varepsilon_{xy}`.  The material
axes are assumed to coincide with the mesh axes, so a laminate whose fibres
run at an angle needs stiffnesses already rotated into the mesh frame.

Giving a non-zero ``stiffness_c11`` is what selects the orthotropic branch:
the four reduced stiffnesses are then used as given and ``youngs_modulus`` and
``poissons_ratio`` are ignored, while a zero value leaves the isotropic
stiffness of the chosen formulation in force.  There is no separate switch, so
a model intended as orthotropic that leaves :math:`c_{11}` unset silently
becomes isotropic with whatever :math:`E` and :math:`\nu` are set.  The branch
carries three limitations, each of which can change an answer without
producing an error message.

* The reduced stiffnesses apply to ``plane_stress`` and ``plane_strain`` only,
  and both then give the same matrix, because the constants are taken as
  supplied rather than derived from an out-of-plane condition; the caller
  supplies reduced plane-stress or reduced plane-strain constants, whichever
  the problem needs.
* Supplying them with ``three_dimensional`` is reported as an error, because a
  full anisotropic stiffness cannot be built from four in-plane numbers.
* Supplying them with ``axisymmetric`` is accepted and then ignored without
  warning, because that formulation always builds its stiffness from :math:`E`
  and :math:`\nu`.

One further consequence deserves stating plainly: in the orthotropic branch
the out-of-plane row of the stiffness is left at zero, so the plane-strain
:math:`\sigma_{zz}` is not recovered and slot 2 of the ``stress`` property
reports zero whatever the in-plane strains are.  The in-plane solution is
unaffected, that row never entering equilibrium, but post-processing that uses
:math:`\sigma_{zz}`, such as a three-dimensional yield or failure measure,
will be wrong there.

Thermal strain
--------------

A temperature change makes an unconstrained body expand without producing
stress; stress arises only from the part of the strain not accounted for by
that free expansion.  With :math:`\alpha` the coefficient of thermal
expansion per degree, :math:`T` the temperature field and
:math:`T_{\mathrm{ref}}` the temperature at which the body is free of stress,
the thermal strain is :math:`\varepsilon^{\mathrm{th}} = \alpha\,(T -
T_{\mathrm{ref}})` and :eq:`elasticity-constitutive` is replaced by

.. math::
   :label: elasticity-thermal

   \sigma_I = \sum_{J=0}^{2} C_{IJ}
              \left( \varepsilon_J - \varepsilon^{\mathrm{th}} \right)
            + \sum_{J=3}^{5} C_{IJ}\, \varepsilon_J .

The thermal part is subtracted from the three normal strain components only,
slots 0, 1 and 2, and the shear components are left untouched.  That is the
correct statement for an isotropic expansion, which changes lengths equally in
all directions and produces no shear, and writing it this way keeps the
thermal strain isotropic even when the stiffness is orthotropic.  The sign
follows from :eq:`elasticity-thermal`: heating a fully restrained body above
:math:`T_{\mathrm{ref}}` makes :math:`\varepsilon_J -
\varepsilon^{\mathrm{th}}` negative and puts the body into compression, as it
should.  Columns of the stiffness that are zero contribute nothing to the
thermal term, so in plane strain, where the :math:`zz` column of the in-plane
rows is zero, the in-plane thermal stress is :math:`-(c_{11} +
c_{12})\,\alpha\,(T - T_{\mathrm{ref}})` with :math:`c_{11} = f(1-\nu)` and
:math:`c_{12} = f\nu`; a reader comparing against a textbook plane-strain
thermoelastic formula that also carries the out-of-plane column should be
aware of the difference.  Thermal strain is applied only when
``thermal_expansion_coefficient`` and ``temperature`` are both set; giving one
without the other is accepted and does nothing, and the default
``reference_temperature`` of zero turns the whole temperature field into a
thermal load, which is seldom what is meant.

The ``temperature`` parameter names an ordinary variable of the problem rather
than a separate field, which has a useful consequence.  If that variable is
itself being solved for, by a heat conduction kernel elsewhere in the same
problem, the thermal stress depends on an unknown of the system; because the
material is written in the automatic differentiation type, the derivative of
the stress with respect to the temperature degrees of freedom is carried
through :eq:`elasticity-thermal` alongside the derivatives with respect to the
displacements.  The thermo-mechanical coupling is therefore differentiated
exactly and Newton's method converges at its usual rate on the fully coupled
system, with nothing lagged or iterated by hand.  A prescribed temperature
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

``DirichletBC`` prescribes the primary variable, :math:`u_i = g(\mathbf{x},
t)` at every node of the named boundary, replacing that node's equilibrium
equation.  A zero value is a rigid support in that direction and a non-zero
value a prescribed settlement or stretch; prescribing one component and
leaving the other free is a roller support, and prescribing all of them is a
clamp.  By default a prescribed displacement is held fixed while loads are
ramped through load steps, and ``scale_with_load=True`` makes it ramp with
them instead, as a displacement-controlled test requires.

Prescribed traction
^^^^^^^^^^^^^^^^^^^

``TractionBC`` prescribes the secondary variable directly: the component of
the surface traction, in force per unit area, conjugate to the variable the
object is attached to.  Because that variable determines the component, the
object has no ``component`` parameter and is added once per displacement
variable on the loaded boundary; the value may be a constant or a function of
position and time.  The sign convention is tied to the coordinate axes and not
to the surface, so a positive value acts along the positive direction of that
variable's axis whichever way the outward normal points, and one positive
value pulls on a right-hand edge while pushing on a left-hand edge.  The value
is multiplied by the thickness before it enters the residual.

Pressure
^^^^^^^^

``PressureBC`` applies a load that does follow the surface.  A fluid pressure
:math:`p` acts normal to the surface and towards the body, so the traction it
exerts is

.. math::
   :label: elasticity-pressure

   t_i = -p\, n_i ,

with :math:`\mathbf{n}` the outward unit normal.  The minus sign is the whole
of the convention: a positive pressure presses inward, against the outward
normal, and a negative value is a suction.  Because each component of the
normal multiplies the pressure separately, this object does take a
``component`` parameter, and one instance is added per displacement variable
with its own component, as for the kernels.  The component must agree with the
component the object's ``variable`` represents; a mismatch is not detected and
applies the pressure along the wrong axis.  The pressure is multiplied by the
thickness in plane problems.

Concentrated force
^^^^^^^^^^^^^^^^^^

``PointSource`` applies a concentrated force at one or more nodes, either by
naming a boundary whose nodes all receive the load or by giving coordinates
that are snapped to the nearest node.  The magnitude is a force, in the units
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

   Q_I = \int_{\partial CD_I \cap \partial\Omega} h\, t_i \,\mathrm{d}S ,

which is the force the support transmits to the body at that node, positive
along the positive direction of the variable's axis.  This is what
:meth:`dualmesh.Problem.reactions` returns node by node and the quantity to
compare against a hand-calculated support reaction; summing over a boundary
with :meth:`dualmesh.Problem.total_reaction` gives the resultant support force
there, which for a body in equilibrium balances the applied loading.

A warning belongs here, because the mistake is natural and its result looks
plausible.  The reaction stored at a node is the whole of the support force at
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
equation, which is to say the flux :eq:`elasticity-flux` assembled by
``StressDivergence``, the traction assembled by ``TractionBC`` and the
pressure assembled by ``PressureBC``, converting stresses per unit area into
forces per unit length of the cross-section so that every residual has the
dimensions of a force.  It applies to ``plane_stress`` and ``plane_strain``
only: in three dimensions there is no out-of-plane dimension to account for,
and in the axisymmetric case the measure :math:`2\pi r` already supplies the
circumferential extent of the material surrounding each control domain, so in
both cases the value must be left at its default of one.  Another value is
accepted there and produces a wrong answer with no diagnostic.

The consistency requirement is the point to remember.  Because :math:`h`
multiplies both sides of the balance, the value given to every
``StressDivergence`` kernel and the value given to every ``TractionBC`` and
``PressureBC`` of the same model must be identical.  A consistent thickness
scales stiffness and distributed loading alike and so cancels out of the
displacement solution of a linear problem; an inconsistent one produces a
model whose loading is scaled by one thickness and whose stiffness is scaled
by another, which changes the answer by their ratio and is difficult to spot
afterwards.  The ``add_plane_elasticity`` helper removes the risk for the
kernels and the body force by passing one value to all of them, but boundary
conditions are added separately and must be given the same value by hand.

One load is deliberately outside this rule.  A ``PointSource`` is **not**
scaled by the thickness, because a concentrated load is a force and not a
force per unit area, so the value supplied must already be the total force
acting through the whole thickness: a line load of :math:`q` per unit length
along the out-of-plane direction becomes a point force of :math:`q h`.

Verification
------------

The module is verified against Chapter 9 of the book in
``tests/python/test_solid_mechanics.py``.  Example 9.8.1, a quadrant of a
plate under uniform edge stress, has the exact solution :math:`u(a) = t_0 a /
E`, which the computed edge displacement matches to machine precision with an
exactly uniform stress field.  Example 9.8.2, a cantilevered plate under a
uniform edge load, reproduces the dual mesh rows of Table 9.8.1 on meshes of
one, two, four and eight elements a side, both for the corner displacements
and for the stress at the centre of the element nearest the support.  The
thick-walled cylinder under internal pressure of Section 9.9.4.1, solved as a
plane-strain quadrant of an annulus with symmetry conditions on the straight
edges and ``PressureBC`` on the bore, reproduces the quadrilateral columns of
Table 9.9.1 to about half a percent, which is the accuracy the book's
statement of the mesh permits, converges monotonically to the analytical
solution of Eq. (9.9.24) on quadrilateral and triangular meshes alike, and
agrees to within a fifth of a percent with the same cylinder solved as an
axisymmetric problem.
