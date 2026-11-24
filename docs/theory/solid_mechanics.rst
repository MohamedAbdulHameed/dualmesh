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
care.  Young's modulus and Poisson's ratio have no numerical default.  When
either is not given, it is the property of the same name that a property
object declares, and the solve stops with an error when no object declares it.
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

Anisotropic elasticity
----------------------

An orthotropic material takes :math:`E_1, E_2, E_3`, :math:`\nu_{12}, \nu_{13},
\nu_{23}` and :math:`G_{12}, G_{13}, G_{23}` in its own axes, through the
compliance

.. math::

   S = \begin{bmatrix}
   1/E_1 & -\nu_{12}/E_1 & -\nu_{13}/E_1 \\
   -\nu_{12}/E_1 & 1/E_2 & -\nu_{23}/E_2 \\
   -\nu_{13}/E_1 & -\nu_{23}/E_2 & 1/E_3
   \end{bmatrix} \oplus \operatorname{diag}(1/G_{23}, 1/G_{13}, 1/G_{12}),
   \qquad C = S^{-1},

which must be positive definite (a stable material).  A fully anisotropic
material gives :math:`C` itself.  ``orientation`` gives the axes 1 and 2 in the
global axes (an angle about :math:`z` in two dimensions), and with :math:`R` the
rotation whose columns are the material axes the stiffness in the global axes is
:math:`C_g = M C M^{\mathsf{T}}`, with :math:`M` the Bond matrix of
:math:`\sigma_g = M \sigma` (Voigt order, tensor shear stresses, engineering
shear strains).  The test checks it against the rotation of the fourth-order
tensor, :math:`C_{ijkl} = R_{ip} R_{jq} R_{kr} R_{ls} C_{pqrs}`.  In plane stress
:math:`\varepsilon_{zz}` follows from :math:`\sigma_{zz} = 0`.

Plasticity and creep
--------------------

**J2 plasticity.**  The yield function is :math:`q(\boldsymbol{\xi}) -
\sigma_y(\bar{\varepsilon}_p, T) \le 0`, with :math:`q` the von Mises stress of
the relative stress :math:`\boldsymbol{\xi} = \mathbf{s} - \sum_k
\boldsymbol{\alpha}_k`, the flow is normal to it, and each back stress follows
Armstrong and Frederick, :math:`\mathrm{d}\boldsymbol{\alpha}_k = \tfrac{2}{3}
C_k\, \mathrm{d}\boldsymbol{\varepsilon}_p - \gamma_k \boldsymbol{\alpha}_k\,
\mathrm{d}\bar{\varepsilon}_p` (Chaboche with several, Prager with
:math:`\gamma_k = 0`).  ``yield_stress`` is any expression of the equivalent
plastic strain and the temperature (linear, Swift, Voce, Ludwik hardening).
Backward Euler with :math:`a_k = 1/(1 + \gamma_k \Delta p)` makes the relative
stress at the end of the step parallel to :math:`\tilde{\boldsymbol{\xi}} =
\mathbf{s}^{\text{trial}} - \sum_k a_k \boldsymbol{\alpha}_k^n`, so the return
is one scalar equation,

.. math::

   q(\tilde{\boldsymbol{\xi}}) - \bigl(3G + \textstyle\sum_k a_k C_k\bigr)\Delta p
   - \sigma_y(\bar{\varepsilon}_p^n + \Delta p) = 0 ,

solved by Newton's method in a bracket.  The derivatives of :math:`\Delta p`
with respect to the unknowns follow from the implicit function theorem, so the
tangent is exact.  With creep, both flow along the deviator of the trial stress,
and the creep and plastic increments solve the two equations :math:`\Delta c =
\Delta t\, \dot{\varepsilon}_c(q_e)` and :math:`q_e = \sigma_y` together, with
:math:`q_e = q^{\text{trial}} - 3G(\Delta c + \Delta p)`.  The tests compare a
strain cycle of Armstrong-Frederick and Chaboche hardening with the uniaxial law
integrated by hand (to :math:`5 \times 10^{-14}`, with the Bauschinger effect),
and monotonic hardening with its closed form at first order in the step.

**Anisotropic plasticity and creep.**  With an anisotropic stiffness or Hill's
equivalent stress :math:`q = \sqrt{\boldsymbol{\sigma} \cdot P
\boldsymbol{\sigma}}` (F, G, H, L, M, N in the material axes), the return is the
closest-point projection: Newton's method on the stress, :math:`\Delta c` and
:math:`\Delta p` at the point, with :math:`S\boldsymbol{\sigma} -
\boldsymbol{\varepsilon}^{\text{trial}} + (\Delta c + \Delta p)\,
\partial q/\partial \boldsymbol{\sigma} = 0`, and the derivatives again by the
implicit function theorem.  For the isotropic stiffness and
:math:`F = G = H = 1/2`, :math:`L = M = N = 3/2` it gives the radial return to
round-off.

**Finite strain.**  The rotated small strain of Hughes and Winget rotates the
elastic strain and the back stresses with the rotation increment, so the history
is objective; the return is that of small strain.  A neo-Hookean material has no
creep or plasticity.

Viscoelasticity, contact and fracture
-------------------------------------

**Viscoelasticity.**  A generalized Maxwell material adds branches to the
long-term moduli.  Each shear branch :math:`(G_i, \tau_i)` carries the deviatoric
stress :math:`\mathbf{s}_i`, which relaxes as :math:`\dot{\mathbf{s}}_i = 2 G_i
\dot{\mathbf{e}} - \mathbf{s}_i / \tau_i`.  With the strain rate constant over
a step, the update is exact:

.. math::

   \mathbf{s}_i^{n+1} = e^{-x} \mathbf{s}_i^n + 2 G_i \frac{1 - e^{-x}}{x}
   \Delta \mathbf{e} , \qquad x = \Delta t / \tau_i ,

and the bulk branches act on the volumetric strain in the same way.  A steady
solve is the relaxed state, with the long-term moduli only, and a step of zero
length is the instantaneous response.

**Contact.**  The mortar contact of two surfaces holds the weighted gap
:math:`g_j = \int N_j (\mathbf{x}_p - \mathbf{x}_s) \cdot \mathbf{n}` of each
secondary node and the contact pressure :math:`\lambda_j`, a Lagrange multiplier,
to :math:`g_j \ge 0`, :math:`\lambda_j \ge 0`, :math:`\lambda_j g_j = 0`.
Newton's method solves the complementarity function :math:`\lambda_j - \max(0,
\lambda_j - c\, g_j / D_j)` with the active set; the scale :math:`c` changes the
convergence only.  Coulomb friction is the complementarity function of Hüeber,
Stadler and Wohlmuth on the weighted slip.

**Cohesive zones.**  A cohesive zone bonds two surfaces that coincide at the
start.  The points of the two surfaces are paired once, on the undeformed
surfaces, as a bonded interface pairs its material points.  The separation of a
pair, :math:`\boldsymbol{\delta} = \mathbf{x}_p - \mathbf{x}_s`, has the normal
part :math:`g_n` and the tangential part :math:`g_t`, and the traction on the
secondary surface is

.. math::

   \mathbf{t} = K (1 - D) \boldsymbol{\delta} ,

except that a closing normal part, :math:`g_n < 0`, keeps the full stiffness
:math:`K`.  The damage grows with the largest effective opening reached,
:math:`\kappa = \max \sqrt{\langle g_n \rangle^2 + g_t^2}`:

.. math::

   D = \frac{\delta_f (\kappa - \delta_0)}{\kappa (\delta_f - \delta_0)} ,
   \qquad \delta_0 = \sigma_c / K , \qquad \delta_f = 2 G_c / \sigma_c ,

with :math:`D = 0` below :math:`\delta_0` and :math:`D = 1` above
:math:`\delta_f`.  The traction peaks at the strength :math:`\sigma_c`.  The
dissipation is :math:`\tfrac{1}{2} K \kappa^2\, \mathrm{d}D = \tfrac{1}{2} K
\delta_0 \delta_f\, \mathrm{d}\kappa`, so the work of separation is
:math:`\tfrac{1}{2} \sigma_c \delta_f = G_c` along any path, in tension, in
shear or mixed.  The physics sets :math:`K = 50 E / h`, with :math:`h` the mean
side of the secondary surface, so that the zone opens by a small fraction of an
element before it softens.  The bodies must be stiffer than the softening slope
:math:`\sigma_c^2 / 2 G_c` for the separation to be stable; otherwise the zone
fails abruptly and the energy it releases is lost from a static solve.

**Fracture mechanics.**  The J-integral of a crack is computed as the domain
integral over a ring around the tip, with a weight that falls from 1 to 0
across the ring, and the stress intensity factors as the interaction integral
with the Williams fields of modes I and II, :math:`K = E' I / 2`.

**Phase-field fracture.**  The AT2 model degrades the tensile part of the strain
energy by :math:`(1 - d)^2`, with the split of Miehe by the sign of the principal
strains (a zero strain is compressive), and solves

.. math::

   -G_c l \nabla^2 d + \left(\frac{G_c}{l} + 2H\right) d = 2H ,
   \qquad H = \max_{\tau \le t} \psi^+ ,

so that the crack costs the energy :math:`G_c` per area and does not heal.

Trusses and shells
------------------

**Trusses.**  A bar of a line mesh, of direction :math:`\mathbf{t}`, carries the
axial force :math:`N = EA\, \varepsilon` with the stretch :math:`\varepsilon =
\mathbf{t} \cdot \partial \mathbf{u} / \partial s`, and nothing across it.  The
equation of the displacement :math:`u_i` takes the flux :math:`N t_i \mathbf{t}`.
A bar has no stiffness across it, so a bar meshed with more than one element has
free interior nodes: a truss takes one element per bar.

**Shells.**  Each quadrilateral of a surface mesh is a flat shell with its own
axes :math:`\mathbf{e}_1, \mathbf{e}_2` in its plane and :math:`\mathbf{e}_3`
normal to it.  The unknowns are the displacements and the rotations
:math:`\boldsymbol{\theta}` about the global axes; in the axes of the facet they
are :math:`\mathbf{u}_l = R \mathbf{u}` and :math:`\boldsymbol{\theta}_l = R
\boldsymbol{\theta}`, and the rotations of the normal are :math:`\beta =
(\theta_{l2}, -\theta_{l1})`.  The strains are those of the Mindlin plate,

.. math::

   \varepsilon_{ab} = \operatorname{sym} \partial_a u_{lb}, \qquad
   \kappa_{ab} = \operatorname{sym} \partial_a \beta_b, \qquad
   \gamma_a = \partial_a w_l + \beta_a ,

with the resultants :math:`N = A[(1-\nu)\varepsilon + \nu \operatorname{tr}
\varepsilon\, I]`, :math:`M = D[(1-\nu)\kappa + \nu \operatorname{tr}\kappa\,
I]` and :math:`Q = k G h \gamma`, :math:`A = Eh/(1-\nu^2)`, :math:`D = A h^2/12`.
The rotation about the normal has no stiffness of its own in a plate, but where
facets meet at an angle it carries bending.  It is tied to the rotation of the
membrane, which it is in the continuum [HughesBrezzi1989]_:
:math:`\omega = \theta_{l3} - \tfrac{1}{2}(\partial_1 u_{l2} - \partial_2
u_{l1})`, with the moment :math:`m = G h\, \omega`.  The virtual work gives the
flux and the source of each global equation.  The shear is integrated at the
centre of the element, as for the plate, so that thin shells do not lock.  On a
flat mesh the shell is the plate formulation, and the tests check it to
:math:`10^{-9}`, also with the plate turned in space.  The Scordelis-Lo roof and
the pinched cylinder [MacNealHarder1985]_ converge from below to their
references: 0.927, 0.969 and 0.990 of 0.3024 on 4, 8 and 16 elements a side, and
0.73, 0.93 and 0.997 of 1.8248e-5 on 8, 16 and 32.


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
all of them is a clamp.  When a solve ramps its data to converge, a
prescribed displacement moves from its starting value to its own with the
loads, as a displacement-controlled test does.

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
along the positive direction of the variable's axis.  This is the quantity to
compare against a hand-calculated support reaction.  Summed over a boundary it
is the resultant support force there,
:meth:`~dualmesh.physics.SolidMechanics.reaction_force`, which for a body in
equilibrium balances the applied loading.

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
``traction``, ``pressure`` and ``mortar_contact`` conditions of a displacement
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
